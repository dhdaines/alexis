"""Classification des unités de texte en format CSV"""

import itertools
import logging
import re
from enum import Enum
from collections.abc import Iterable, Sequence
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
    line = []
    for word, xdelta in zip(paragraph, xdeltas):
        if xdelta < 0:
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
    chapitre: Optional[str] = None
    en_tete: Optional[str] = None
    article_idx: int = 0
    in_toc: bool = False
    debut_chapitre: bool = False

    def classify_alinea(
        self, tag: str, paragraph: Sequence[dict[str, Any]], text: str
    ) -> str:
        if tag == "Tete":
            # enregistrer l'en-tete pour assister en classification
            en_tete = text.lower()
            if en_tete != self.en_tete:
                self.en_tete = en_tete
                self.debut_chapitre = True
            else:
                self.debut_chapitre = False
            return tag
        elif tag == "Pied":
            return tag

        word = paragraph[0]["text"].lower()
        if m := re.match(r"article (\d+)", text, re.IGNORECASE):
            tag = "Article"
            self.article_idx = int(m.group(1))
        elif m := re.match(r"(\d+)[\)\.]?", text):
            idx = int(m.group(1))
            if idx == self.article_idx + 1:
                self.article_idx = idx
                tag = "Article"
            else:
                tag = "Enumeration"
        elif re.match(r"[a-z][\)\.]|[•-]", text):
            tag = "Enumeration"
        elif word == "attendu":
            tag = "Attendu"
            if re.match(r".*avis de motion", text, re.IGNORECASE):
                tag = "Avis"
        elif word == "chapitre":
            # voyons donc (faut du machine learning chose)
            if (
                self.chapitre is not None
                and "dispositions déclaratoires" in self.chapitre
                and int(paragraph[0]["top"]) > 200
            ):
                pass
            else:
                tag = "Chapitre"
                self.chapitre = text.lower()
        elif word == "section":
            tag = "Section"
        elif word == "sous-section":
            # voyons donc #2
            if (
                self.chapitre is not None
                and "dispositions déclaratoires" in self.chapitre
                and int(paragraph[0]["top"]) > 200
            ):
                pass
            else:
                tag = "SousSection"
        elif text.isupper() and int(paragraph[0]["x0"]) == 193:
            # voyons donc #3 (problème de pdfplumber...?)
            tag = "SousSection"
        elif text.isupper() and self.debut_chapitre:
            # voyons donc #4 (problème de pdfplumber...?)
            tag = "Chapitre"
        elif (
            re.match(
                r"r[eè]glement ?(?:de|d'|sur|relatif aux)?",
                text,
                re.IGNORECASE,
            )
            and int(paragraph[0]["page"]) < 5
        ):
            tag = "Titre"

        return tag

    def classify_paragraph_heuristic(
        self, tag: str, paragraph: Sequence[dict[str, Any]]
    ) -> Iterable[tuple[str, Sequence[dict[str, Any]]]]:
        """Classification heuristique très simplistique d'un alinéa, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption.
        """
        if len(paragraph) == 0:
            return []
        text = " ".join(w["text"] for w in paragraph)

        if re.match("^table des mati[èe]res", text, re.IGNORECASE):
            self.in_toc = True
        if self.in_toc:
            # FIXME: Not entirely reliable way to detect end of TOC
            if tag == "Tete" and re.match(".*chapitre", text, re.IGNORECASE):
                self.in_toc = False
            elif tag not in ("Pied", "Tete"):
                return [("TOC", paragraph)]

        # Detecter les dates dans des tableaux
        if tag == "Tableau" and "avis de motion" in text.lower():
            return extract_dates(paragraph)

        tag = self.classify_alinea(tag, paragraph, text)

        if tag == "Enumeration":
            return extract_enumeration(paragraph)
        return [(tag, paragraph)]

    def output_paragraph(
        self, tag: str, paragraph: Sequence[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        word = paragraph[0]
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

    def classify_heuristic(
        self, words: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Classification heuristique très simplistique des alinéas, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption."""
        for tag, paragraph in group_paragraphs(words):
            if not paragraph:
                return
            # FIXME: Can split but not join paragraphs
            for t, p in self.classify_paragraph_heuristic(tag, paragraph):
                yield from self.output_paragraph(t, p)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        words = self.classify_heuristic(words)
        return words
