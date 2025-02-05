"""Conversion de PDF en CSV"""

import csv
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, TextIO

from pdfplumber import PDF
from pdfplumber.page import Page
from pdfplumber.structure import PDFStructElement, PDFStructTree, StructTreeMissing

from .types import T_obj

LOGGER = logging.getLogger("convert")
FIELDNAMES = [
    "sequence",
    "segment",
    "text",
    "page",
    "page_width",
    "page_height",
    "fontname",
    "rgb",
    "x0",
    "x1",
    "top",
    "bottom",
    "doctop",
    "mcid",
    "mctag",
    "tagstack",
]


def write_csv(
    doc: Iterable[dict[str, Any]], outfh: TextIO, fieldnames: list[str] = FIELDNAMES
):
    writer = csv.DictWriter(outfh, fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(doc)


def get_rgb(c: T_obj) -> str:
    """Extraire la couleur d'un objet en chiffres hexadécimaux"""
    couleur = c.get("non_stroking_color", c.get("stroking_color"))
    if couleur is None or couleur == "":
        return "#000"
    elif len(couleur) == 1:
        return "#" + "".join(
            (
                "%x" % int(min(0.999, val) * 16)
                for val in (couleur[0], couleur[0], couleur[0])
            )
        )
    elif len(couleur) == 3 or len(couleur) == 4:
        # Could be RGB, RGBA, CMYK...
        return "#" + "".join(("%x" % int(min(0.999, val) * 16) for val in couleur))
    else:
        LOGGER.warning("Espace couleur non pris en charge: %s", couleur)
        return "#000"


def get_word_features(
    word: T_obj,
    page: Page,
    chars: dict[tuple[int, int], T_obj],
    elmap: dict[int, str],
) -> T_obj:
    # Extract things from first character (we do not use
    # extra_attrs because otherwise extract_words will
    # insert word breaks)
    feats = word.copy()
    if c := chars.get((word["x0"], word["top"])):
        feats["rgb"] = get_rgb(c)
        mcid = c.get("mcid")
        if mcid is not None:
            feats["mcid"] = mcid
            if mcid in elmap:
                feats["tagstack"] = elmap[mcid]
        feats["mctag"] = c.get("tag")
        feats["fontname"] = c.get("fontname")
    # Ensure matching PDF/CSV behaviour with missing fields
    for field in "mcid", "sequence", "segment", "fontname", "tagstack":
        if field not in feats or feats[field] is None:
            feats[field] = ""
    feats["page"] = page.page_number
    feats["page_height"] = round(float(page.height))
    feats["page_width"] = round(float(page.width))
    # Round positions to points
    for field in "x0", "x1", "top", "bottom", "doctop":
        feats[field] = round(float(word[field]))
    return feats


class Converteur:
    pdf: PDF
    path: Path
    tree: Optional[PDFStructTree]
    y_tolerance: int

    def __init__(
        self,
        path: Path,
        y_tolerance: int = 2,
    ):
        self.pdf = PDF.open(path)
        self.path = path
        self.y_tolerance = y_tolerance
        try:
            # Get the tree for the *entire* document since elements
            # like the TOC may span multiple pages, and we won't find
            # them if we look at the parent tree for other than the
            # page in which the top element appears (this is the way
            # the structure tree implementation in pdfplumber works,
            # which might be a bug)
            self.tree = PDFStructTree(self.pdf)
        except StructTreeMissing:
            self.tree = None

    def element_map(self, page_number: int) -> dict[int, str]:
        """Construire une correspondance entre MCIDs et types d'elements structurels"""
        elmap: dict[int, str] = {}
        if self.tree is None:
            return elmap
        d: deque[PDFStructElement | str] = deque(self.tree)
        tagstack: deque[str] = deque()
        while d:
            el = d.pop()
            if isinstance(el, str):
                assert tagstack[-1] == el
                tagstack.pop()
            else:
                d.append(el.type)
                tagstack.append(el.type)
                if el.page_number == page_number:
                    for mcid in el.mcids:
                        elmap[mcid] = ";".join(tagstack)
                d.extend(el.children)
        return elmap

    def extract_words(self, pages: Optional[Iterable[int]] = None) -> Iterator[T_obj]:
        if pages is None:
            pages = range(1, len(self.pdf.pages) + 1)
        for idx in pages:
            page = self.pdf.pages[idx - 1]
            LOGGER.info("traitement de la page %d", page.page_number)
            words = page.extract_words(y_tolerance=self.y_tolerance)
            elmap = self.element_map(page.page_number)
            # Index characters for lookup
            chars = dict(((c["x0"], c["top"]), c) for c in page.chars)
            for word in words:
                if word["x0"] < 0 or word["top"] < 0:
                    continue
                if word["x1"] > page.width or word["bottom"] > page.height:
                    continue
                feats = get_word_features(word, page, chars, elmap)
                feats["path"] = str(self.path)
                yield feats
            page.close()
