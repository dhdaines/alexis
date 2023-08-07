"""Classification des unités de texte en format CSV"""

import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Iterator, Optional

LOGGER = logging.getLogger("label")


def group_paragraphs(
    words: Iterable[dict[str, Any]]
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Grouper le flux de mots par unité (BI*) de texte."""
    bio = tag = "O"
    paragraph = []
    for word in words:
        next_bio, _, next_tag = word["tag"].partition("-")
        if not next_tag:
            next_tag = next_bio
        if next_bio != bio or next_bio == "B":
            if next_bio == "I":
                paragraph.append(word)
            else:
                # Could do this without storing the paragraph, but not
                # strictly necessary (it's not that big!)
                if paragraph:
                    yield tag, paragraph
                paragraph = [word]
                # Note: tag *could* change inside a block
                tag = next_tag
            bio = next_bio
        else:
            paragraph.append(word)
    if paragraph:
        yield tag, paragraph


def line_breaks(
    paragraph: Sequence[dict[str, Any]]
) -> Iterable[Sequence[dict[str, Any]]]:
    xdeltas = [int(paragraph[0]["x0"])]
    xdeltas.extend(
        int(b["x0"]) - int(a["x0"]) for a, b in itertools.pairwise(paragraph)
    )
    ydeltas = [int(paragraph[0]["top"])]
    ydeltas.extend(
        int(b["top"]) - int(a["top"]) for a, b in itertools.pairwise(paragraph)
    )
    line = []
    for word, xdelta, ydelta in zip(paragraph, xdeltas, ydeltas):
        if xdelta <= 0 and ydelta > 0:  # CR, LF
            yield line
            line = []
        line.append(word)
    if line:
        yield line


def extract_dates(
    paragraph: Sequence[dict[str, Any]]
) -> Iterable[tuple[str, Sequence[dict[str, Any]]]]:
    for line in line_breaks(paragraph):
        text = " ".join(w["text"] for w in line).lower()
        if "avis" in text:
            tag = "Avis"
        elif "vigueur" in text:
            tag = "Vigueur"
        elif "projet" in text:  # Doit venir avant "adoption"
            tag = "Projet"
        elif "adoption" in text:
            tag = "Adoption"
        elif "mrc" in text:
            tag = "MRC"
        elif "consultation publique" in text:
            tag = "Publique"
        elif "consultation écrite" in text:
            tag = "Ecrite"
        yield tag, line


class Bullet(Enum):
    NUMERIC = re.compile(r"(\d+)[\)\.]?")
    ALPHABETIC = re.compile(r"([a-z])[\)\.]", re.IGNORECASE)
    ROMAN = re.compile(r"([xiv]+)[\)\.]", re.IGNORECASE)
    BULLET = re.compile(r"([•-])")


def extract_enumeration(
    paragraph: Sequence[dict[str, Any]]
) -> Iterable[tuple[str, Sequence[dict[str, Any]]]]:
    word = paragraph[0]
    pattern = None
    for pattern in Bullet:
        if pattern.value.match(word["text"]):
            break
    if pattern is None:
        return [("Enumeration", paragraph)]
    item = [word]
    for word in paragraph[1:]:
        if pattern.value.match(word["text"]):
            if item:
                yield ("Enumeration", item)
                item = []
        item.append(word)
    if item:
        yield ("Enumeration", item)


class Classificateur:
    head_chapitre: Optional[str] = None
    article_idx: int = 0
    in_toc: bool = False
    prev_x0: int = 72

    def classify_alinea(
        self,
        tag: str,
        paragraph: Sequence[dict[str, Any]],
        text: str,
        page_words: Sequence[dict[str, Any]],
    ) -> str:
        if tag in ("Tete", "Pied"):
            return tag
        word = paragraph[0]["text"].lower()
        if m := re.match(r"article (\d+)", text, re.IGNORECASE):
            tag = "Article"
            self.article_idx = int(m.group(1))
        elif m := re.match(r"(?!\d+[\)\.]).*\n([1-9]\d*)[\)\.](?!\d)", text):
            tag = "Article"
            self.article_idx = int(m.group(1))
        elif m := re.match(r"(\d+)[\)\.]?", text):
            idx = int(m.group(1))
            # FIXME: Bad heuristic! Bad!
            if idx == self.article_idx + 1:
                self.article_idx = idx
                tag = "Article"
            else:
                tag = "Enumeration"
        elif re.match(r"[a-z][\)\.]|[•-]", text):
            tag = "Enumeration"
        elif m := re.match(r"figure\s+(\d+)", text, re.IGNORECASE):
            tag = "Figure"
        elif word == "attendu":
            tag = "Attendu"
            if re.match(r".*avis de motion", text, re.IGNORECASE):
                tag = "Avis"
        elif word == "chapitre" and paragraph[1]["text"] != "X":
            tag = "Chapitre"
        elif word == "annexe" and paragraph[1]["text"] != "X":
            tag = "Annexe"
        elif word == "section" and paragraph[1]["text"] != "X":
            tag = "Section"
        elif word == "sous-section" and paragraph[1]["text"] != "X.X":
            tag = "SousSection"
        elif (
            re.match(
                r"(?:ville de sainte-adèle.*)?r[eè]glement.*(?:de|d'|sur|relatif aux?|concernant|numero|numéro|no\.)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            and int(paragraph[0]["page"]) < 3
        ):
            tag = "Titre"
        elif (
            re.match(
                r"^r[eè]glement\s+(?:numero|numéro|no\.\s+)?(\S+)$",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            and int(paragraph[0]["page"]) < 3
        ):
            tag = "Titre"
        elif (
            re.match(
                r"(?:afin de|sur|relatif aux?|concernant|numero|numéro|no\.)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            and int(paragraph[0]["page"]) < 3
        ):
            tag = "Titre"
        elif text.isupper() or text == "T1.2 Récréation":
            if (
                # Rough heurstic for "SOUS-SECTION N.N" as an image
                any(w["text"].isalpha() for w in paragraph)
                and int(paragraph[0]["x0"]) - int(self.prev_x0) > 100
            ):
                if text != "CHAPITRE X":
                    tag = "SousSection"

        return tag

    def classify_paragraph_heuristic(
        self,
        tag: str,
        paragraph: Sequence[dict[str, Any]],
        page_words: Sequence[dict[str, Any]],
    ) -> Iterable[tuple[str, Sequence[dict[str, Any]]]]:
        """Classification heuristique très simplistique d'un alinéa, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption.
        """
        if len(paragraph) == 0:
            return []
        text = "\n".join(
            " ".join(w["text"] for w in line) for line in line_breaks(paragraph)
        )

        if re.match("^table des mati[èe]res", text, re.IGNORECASE):
            self.in_toc = True

        if tag == "Tete":
            if m := re.match(r".*(chapitre)\s+(\S+)", text, re.IGNORECASE):
                self.head_chapitre = m.group(2)
            else:
                self.head_chapitre = None

        # "CHAPITRE N" is sometimes an image, so check for upper-case text
        # on a page by itself
        n_nonhead_words = sum(
            1
            for w in page_words
            if w["tag"] not in ("B-Tete", "I-Tete", "B-Pied", "I-Pied")
        )
        if (
            text.isupper()
            and "ANNEXE" not in text
            and n_nonhead_words == len(paragraph)
        ):
            self.in_toc = False
            return [("Chapitre", paragraph)]

        if self.in_toc:
            # If we found a chapter name in the header then end the TOC
            if self.head_chapitre is not None:
                self.in_toc = False
            elif tag not in ("Pied", "Tete"):
                return [("TOC", paragraph)]

        if tag == "Tableau":
            # Detecter les dates dans des tableaux.
            # FIXME: diverses autres façons de spécifier les
            # dates... besoin d'un vrai taggeur!
            if "avis de motion" in text.lower():
                return extract_dates(paragraph)
            # Detecter les titres aussi...
            if re.match(
                r"(?:ville de sainte-ad[eè]le\s*)?r[eè]glement\s+(?:relatif|sur|de|\d|d)",
                text,
                re.IGNORECASE,
            ):
                pass
            else:
                # Sinon, les laisser tranquilles!
                return [(tag, paragraph)]

        tag = self.classify_alinea(tag, paragraph, text, page_words)

        if tag == "Enumeration":
            return extract_enumeration(paragraph)
        return [(tag, paragraph)]

    def output_paragraph(
        self, tag: str, paragraph: Sequence[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        word = paragraph[0]
        cur_x0 = word["x0"]
        if tag == "O":
            word["tag"] = "O"
        else:
            word["tag"] = f"B-{tag}"
        yield word
        for word in paragraph[1:]:
            if tag == "O":
                word["tag"] = "O"
            else:
                word["tag"] = f"I-{tag}"
            yield word
        self.prev_x0 = cur_x0

    def classify_heuristic(
        self, words: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Classification heuristique très simplistique des alinéas, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption."""
        for page_number, page_words in itertools.groupby(
            words, key=lambda x: x["page"]
        ):
            page_words_list = list(page_words)
            for tag, paragraph in group_paragraphs(page_words_list):
                if not paragraph:
                    continue
                # FIXME: Can split but not join paragraphs
                for t, p in self.classify_paragraph_heuristic(
                    tag, paragraph, page_words_list
                ):
                    yield from self.output_paragraph(t, p)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        words = self.classify_heuristic(words)
        return words
