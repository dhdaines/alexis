"""Segmentation du texte en format CSV"""

import csv
import itertools
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Iterator, TextIO

LOGGER = logging.getLogger("segment")


def detect_page_margins(
    page_number: int, page_words: Sequence[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Détection heuristique des marges (haut et bas seuelement) d'une
    page, qui sont étiquettées avec 'Tete' et 'Pied'."""
    if not page_words:
        return
    for w in page_words:
        w["top"] = round(float(w["top"]))
        w["bottom"] = round(float(w["bottom"]))
    margin_top = 0
    page_height = round(float(page_words[0]["page_height"]))
    margin_bottom = page_height
    l1top = page_words[0]["top"]
    l1bottom = page_words[0]["bottom"]
    l1size = l1bottom - l1top
    l1 = [word["text"] for word in page_words if word["top"] == l1top]
    letop = page_words[-1]["top"]
    lebottom = page_words[-1]["bottom"]
    lesize = lebottom - letop
    le = [word["text"] for word in page_words if word["top"] == letop]
    if len(le) == 1 and le[0].isnumeric() and len(le[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en pied trouvé à %f pt", page_number, letop
        )
        margin_bottom = letop
    elif lesize < 10 and (page_height - lebottom) < 72:
        LOGGER.info("page %d: pied de page trouvé à %f pt", page_number, letop)
        margin_bottom = letop
        # Il existe parfois deux lignes de pied de page
        for w in page_words[::-1]:
            if w["top"] == letop:
                continue
            wsize = w["bottom"] - w["top"]
            if wsize < 10 and letop - w["bottom"] < 10:
                LOGGER.info(
                    "page %d: deuxième ligne de pied de page trouvé à %f pt",
                    page_number,
                    w["top"],
                )
                margin_bottom = w["top"]
                break
    if len(l1) == 1 and l1[0].isnumeric() and len(l1[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en tête trouvé à %f", page_number, l1bottom
        )
        margin_top = l1bottom
    elif l1size < 10 and l1top < 72:
        LOGGER.info("page %d: en-tête trouvé a %f pt", page_number, l1bottom)
        margin_top = l1bottom
        # Il existe parfois deux lignes de tête de page aussi!
        for w in page_words:
            if w["top"] == l1top:
                continue
            wsize = w["bottom"] - w["top"]
            if wsize < 10 and w["top"] - l1bottom < 10:
                LOGGER.info(
                    "page %d: deuxième ligne d'en-tête trouvé à %f pt",
                    page_number,
                    w["bottom"],
                )
                margin_top = w["bottom"]
                break
    seen_head = seen_foot = False
    for word in page_words:
        if word["bottom"] <= margin_top:
            word["tag"] = "I-Tete" if seen_head else "B-Tete"
            seen_head = True
        elif word["top"] >= margin_bottom:
            word["tag"] = "I-Pied" if seen_foot else "B-Pied"
            seen_foot = True
        yield word


def group_paragraphs(
    words: Iterable[dict[str, Any]]
) -> Iterator[tuple[str, Iterable[dict[str, Any]]]]:
    """Grouper un flux de mots par unité (BI*) de texte."""
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


class Segmenteur:
    def detect_margins(
        self, words: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Détection heuristique des marges (haut et bas seuelement) de chaque
        page, qui sont étiquettées avec 'Tete' et 'Pied'."""
        for page_number, page_words in itertools.groupby(
            words, key=lambda x: x["page"]
        ):
            yield from detect_page_margins(page_number, list(page_words))

    def split_paragraphs(
        self, words: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Détection heuristique très simple des alinéas.  Un nouvel alinéa
        est marqué lorsque l'interligne dépasse 1,5 fois la hauteur du
        texte.
        """
        prev_top = 0
        for word in words:
            if word["tag"]:
                pass
            elif word["top"] - prev_top < 0:
                word["tag"] = "B-Alinea"
            elif word["top"] - prev_top >= 1.5 * (word["bottom"] - word["top"]):
                word["tag"] = "B-Alinea"
            else:
                word["tag"] = "I-Alinea"
            prev_top = word["top"]
            yield word

    def classify_paragraph_heuristic(
        self, tag: str, paragraph: Sequence[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Classification heuristique très simplistique d'un alinéa, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption.
        """
        if len(paragraph) == 0:
            return
        word = paragraph[0]
        if word["text"].lower() == "article":
            tag = "Article"
        elif word["text"].lower() == "attendu":
            tag = "Attendu"
            # Un train peut en cacher un autre
            for idx in range(len(paragraph) - 3):
                if [
                    w["text"].lower() for w in paragraph[idx : idx + 3]
                ] == "avis de motion".split():
                    tag = "Avis"
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
            yield from self.classify_paragraph_heuristic(tag, paragraph)

    def __call__(self, infh: TextIO) -> list[dict[str, Any]]:
        reader = csv.DictReader(infh)
        words = self.detect_margins(reader)
        words = self.split_paragraphs(words)
        words = self.classify_heuristic(words)
        return list(words)
