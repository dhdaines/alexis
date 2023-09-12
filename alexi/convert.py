"""Conversion de PDF en CSV"""

import itertools
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.structure import (PDFStructElement, PDFStructTree,
                                  StructTreeMissing)
from pdfplumber.utils.geometry import T_bbox, objects_to_bbox

LOGGER = logging.getLogger("convert")
FIELDNAMES = [
    "tag",
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


def get_tables(tree: PDFStructTree, page_number: int) -> Iterator[PDFStructElement]:
    """Trouver les tableaux principaux dans un page (ignorer les tableaux
    imbriqués)"""
    d = deque(tree)
    while d:
        el = d.popleft()
        if el.type == "Table":
            if el.page_number == page_number:
                yield el
        else:
            # On ne traîte pas des tableaux imbriqués
            d.extend(el.children)


def get_figures(tree: PDFStructTree, page_number: int) -> Iterator[PDFStructElement]:
    """Trouver les figures dans un page (ignorer celles imbriqués dans des
    tableaux ou autres figures)"""
    d = deque(tree)
    while d:
        el = d.popleft()
        if el.type == "Table":
            # Ignorer les figures à l'intérieur d'un tableau
            continue
        elif el.type == "Figure":
            if el.page_number == page_number:
                yield el
        else:
            d.extend(el.children)


def get_child_mcids(el: PDFStructElement) -> Iterator[int]:
    """Trouver tous les MCIDs à l'intérieur d'un élément structurel"""
    yield from el.mcids
    d = deque(el.children)
    while d:
        el = d.popleft()
        yield from el.mcids
        d.extend(el.children)


def bbox_overlaps(obox: T_bbox, bbox: T_bbox) -> bool:
    """Déterminer si deux BBox ont une intersection."""
    ox0, otop, ox1, obottom = obox
    x0, top, x1, bottom = bbox
    return ox0 < x1 and ox1 > x0 and otop < bottom and obottom > top


def bbox_contains(bbox: T_bbox, ibox: T_bbox) -> bool:
    """Déterminer si une BBox est contenu entièrement par une autre."""
    x0, top, x1, bottom = bbox
    ix0, itop, ix1, ibottom = ibox
    return ix0 >= x0 and ix1 <= x1 and itop >= top and ibottom <= bottom


def get_element_bbox(page: Page, el: PDFStructElement) -> T_bbox:
    """Obtenir le BBox autour d'un élément structurel."""
    bbox = el.attributes.get("BBox", None)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        top = page.height - y1
        bottom = page.height - y0
        return (x0, top, x1, bottom)
    else:
        mcids = set(get_child_mcids(el))
        mcid_objs = [
            c
            for c in itertools.chain.from_iterable(page.objects.values())
            if c.get("mcid") in mcids
        ]
        if not mcid_objs:
            return (-1, -1, -1, -1)  # An impossible BBox
        return objects_to_bbox(mcid_objs)


def add_margin(bbox: T_bbox, page: Page, margin: int):
    """Ajouter une marge autour d'un BBox"""
    x0, top, x1, bottom = bbox
    return (
        max(0, x0 - margin),
        max(0, top - margin),
        min(page.width, x1 + margin),
        min(page.height, bottom + margin),
    )


def make_element_map(tree: Optional[PDFStructTree], page_number: int) -> dict[int, str]:
    """Construire une correspondance entre MCIDs et types d'elements structurels"""
    elmap: dict[int, str] = {}
    if tree is None:
        return elmap
    d: deque[PDFStructElement | str] = deque(tree)
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


def get_rgb(c: dict) -> str:
    """Extraire la couleur d'un objet en 3 chiffres hexadécimaux"""
    couleur = c.get("non_stroking_color", c.get("stroking_color"))
    if couleur is None:
        return "000"
    elif len(couleur) == 1:
        r = g = b = couleur[0]
    elif len(couleur) == 3:
        r, g, b = couleur
    else:
        LOGGER.warning("Espace couleur non pris en charge: %s", couleur)
        return "000"
    return "".join(("%x" % int(min(0.999, val) * 16) for val in (r, g, b)))


def get_word_features(
    word: dict,
    page: Page,
    chars: dict[tuple[int, int], dict[str, Any]],
    elmap: dict[int, str],
) -> dict:
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
    for field in "mcid", "tag", "fontname", "tagstack":
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
    imgdir: Optional[Path] = None

    def __init__(self, imgdir=None):
        self.imgdir = imgdir

    def extract_words(
        self, pdf: pdfplumber.PDF, pages: Optional[list[int]] = None
    ) -> Iterator[dict[str, Any]]:
        if pages is None:
            pages = list(range(len(pdf.pages)))
        try:
            # Get the tree for the *entire* document since elements
            # like the TOC may span multiple pages, and we won't find
            # them if we look at the parent tree for other than the
            # page in which the top element appears (this is the way
            # the structure tree implementation in pdfplumber works,
            # which might be a bug)
            tree = PDFStructTree(pdf)
        except StructTreeMissing:
            tree = None
        for idx in pages:
            page = pdf.pages[idx]
            LOGGER.info("traitement de la page %d", page.page_number)
            words = page.extract_words(y_tolerance=2)
            elmap = make_element_map(tree, page.page_number)
            self.save_figures(page, tree)
            # Index characters for lookup
            chars = dict(((c["x0"], c["top"]), c) for c in page.chars)
            for word in words:
                yield get_word_features(word, page, chars, elmap)

    def save_figures(self, page: Page, tree: Optional[PDFStructTree]):
        if tree is None:
            return
        tboxes = [
            get_element_bbox(page, table)
            for table in get_tables(tree, page.page_number)
        ]
        for idx, tbox in enumerate(tboxes):
            if tbox is None:
                continue
            if self.imgdir is not None:
                img = page.crop(add_margin(tbox, page, 10)).to_image(
                    resolution=150, antialias=True
                )
                img.save(self.imgdir / f"page{page.page_number}-table{idx + 1}.png")
        fboxes = [
            get_element_bbox(page, figure)
            for figure in get_figures(tree, page.page_number)
        ]
        for idx, fbox in enumerate(fboxes):
            if fbox is None:
                continue
            in_table = False
            for tbox in tboxes:
                # logical structure doesn't prevent this from happening
                if bbox_overlaps(tbox, fbox):
                    LOGGER.info("Figure in table: %s in %s", fbox, tbox)
                    in_table = True
                    break
            if in_table:
                continue
            if self.imgdir is not None:
                try:
                    img = page.crop(fbox).to_image(resolution=150, antialias=True)
                    fboxtxt = ",".join(str(round(x)) for x in fbox)
                    img.save(
                        self.imgdir / f"page{page.page_number}-figure-{fboxtxt}.png"
                    )
                except ValueError as e:
                    LOGGER.warning(
                        "Failed to save figure on page %d at %s: %s",
                        page.page_number,
                        fbox,
                        e,
                    )

    def __call__(
        self, infh: Any, pages: Optional[list[int]] = None
    ) -> Iterable[dict[str, Any]]:
        with pdfplumber.open(infh) as pdf:
            yield from self.extract_words(pdf, pages)
