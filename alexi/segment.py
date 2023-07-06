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
    seen_head = seen_foot = False
    for word in page_words:
        if word["bottom"] <= margin_top:
            word["tag"] = "I-Tete" if seen_head else "B-Tete"
            seen_head = True
        elif word["top"] >= margin_bottom:
            word["tag"] = "I-Pied" if seen_foot else "B-Pied"
            seen_foot = True
        yield word


def detect_margins(words: Sequence[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for page_number, page_words in itertools.groupby(words, key=lambda x: x["page"]):
        for word in detect_page_margins(page_number, list(page_words)):
            yield word


def split_paragraphs(words: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    prev_top = 0
    for word in words:
        if word["tag"]:
            pass
        elif word["top"] - prev_top < 0:
            word["tag"] = "B"
        elif word["top"] - prev_top >= 1.5 * (word["bottom"] - word["top"]):
            word["tag"] = "B"
        else:
            word["tag"] = "I"
        prev_top = word["top"]
        yield word


class Segmenteur:
    def __call__(self, infh: TextIO) -> list[dict[str, Any]]:
        reader = csv.DictReader(infh)
        words = detect_margins(list(reader))
        words = split_paragraphs(words)
        return list(words)
