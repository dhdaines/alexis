"""Conversion de PDF en CSV"""

import itertools
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pdfplumber
from pdfplumber.utils import obj_to_bbox

LOGGER = logging.getLogger("convert")
FIELDNAMES = [
    "tag",
    "text",
    "page",
    "page_width",
    "page_height",
    "r",
    "g",
    "b",
    "x0",
    "x1",
    "top",
    "bottom",
    "doctop",
    "mcid",
    "mctag",
]


def get_tables(page):
    st = page.structure_tree
    d = deque(st)
    while d:
        el = d.popleft()
        if el["type"] == "Table":
            yield el
        elif "children" in el:
            # On ne traîte pas des tableaux imbriqués
            d.extend(el["children"])


def get_figures(page):
    st = page.structure_tree
    d = deque(st)
    while d:
        el = d.popleft()
        if el["type"] == "Table":
            # Ignorer les figures à l'intérieur d'un tableau
            continue
        if el["type"] == "Figure":
            yield el
        elif "children" in el:
            d.extend(el["children"])


def get_child_mcids(el):
    d = deque([el])
    while d:
        el = d.popleft()
        if "children" in el:
            d.extend(el["children"])
        if "mcids" in el:
            yield from el["mcids"]


def bbox_overlaps(obox, bbox):
    ox0, otop, ox1, obottom = obox
    x0, top, x1, bottom = bbox
    return ox0 < x1 and ox1 > x0 and otop < bottom and obottom > top


def get_element_bbox(page, el):
    mcids = set(get_child_mcids(el))
    mcid_objs = [
        c
        for c in itertools.chain.from_iterable(page.objects.values())
        if c.get("mcid") in mcids
    ]
    if not mcid_objs:
        return None
    return pdfplumber.utils.objects_to_bbox(mcid_objs)


def get_thingy_bbox(page, thingy):
    bbox = thingy.get("attributes", {}).get("BBox", None)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        top = page.height - y1
        bottom = page.height - y0
        return (x0, top, x1, bottom)
    else:
        return get_element_bbox(page, thingy)


def add_margin(bbox, margin):
    x0, top, x1, bottom = bbox
    return (max(0, x0 - margin), max(0, top - margin), x1 + margin, bottom + margin)


class Converteur:
    imgdir: Optional[Path] = None

    def __init__(self, imgdir=None):
        self.imgdir = imgdir

    def extract_words(
        self, pdf: pdfplumber.PDF, pages: Optional[list[int]] = None
    ) -> Iterator[dict[str, Any]]:
        if pages is None:
            pages = list(range(len(pdf.pages)))
        for idx in pages:
            p = pdf.pages[idx]
            words = p.extract_words(y_tolerance=1)
            tables = list(get_tables(p))
            tboxes = [get_thingy_bbox(p, table) for table in tables]
            for idx, tbox in enumerate(tboxes):
                if tbox is None:
                    continue
                if self.imgdir is not None:
                    img = p.crop(add_margin(tbox, 10)).to_image(
                        resolution=150, antialias=True
                    )
                    img.save(self.imgdir / f"page{p.page_number}-table{idx + 1}.png")
            for idx, f in enumerate(get_figures(p)):
                fbox = get_thingy_bbox(p, f)
                if fbox is None:
                    continue
                in_table = False
                # get_figures is supposed to prevent this, but doesn't, it seems
                for tbox in tboxes:
                    if bbox_overlaps(fbox, tbox):
                        in_table = True
                if in_table:
                    continue
                if self.imgdir is not None:
                    try:
                        img = p.crop(fbox).to_image(resolution=150, antialias=True)
                        fboxtxt = ",".join(str(round(x)) for x in fbox)
                        img.save(
                            self.imgdir / f"page{p.page_number}-figure-{fboxtxt}.png"
                        )
                    except ValueError as e:
                        LOGGER.warning(
                            "Failed to save figure on page %d at %s: %s",
                            p.page_number,
                            fbox,
                            e,
                        )

            # Index characters for lookup
            chars = dict(((c["x0"], c["top"]), c) for c in p.chars)
            LOGGER.info("traitement de la page %d", p.page_number)
            prev_table = None
            for w in words:
                # Extract colour from first character (FIXME: assumes RGB space)
                if c := chars.get((w["x0"], w["top"])):
                    if c.get("non_stroking_color") is None:
                        w["r"] = w["g"] = w["b"] = 0
                    elif len(c["non_stroking_color"]) == 1:
                        w["r"] = w["g"] = w["b"] = c["non_stroking_color"][0]
                    elif len(c["non_stroking_color"]) == 3:
                        w["r"], w["g"], w["b"] = c["non_stroking_color"]
                    else:
                        LOGGER.warning(
                            "Espace couleur non pris en charge: %s",
                            c["non_stroking_color"],
                        )
                    w["mcid"] = c.get("mcid")
                    w["mctag"] = c.get("tag")
                w["page"] = p.page_number
                w["page_height"] = round(float(p.height))
                w["page_width"] = round(float(p.width))
                # Find words inside tables and tag accordingly
                for table, tbox in zip(tables, tboxes):
                    if tbox is None:
                        continue
                    if bbox_overlaps(obj_to_bbox(w), tbox):
                        if id(table) != prev_table:
                            w["tag"] = "B-Tableau"
                        else:
                            w["tag"] = "I-Tableau"
                        prev_table = id(table)
                        break

                # Round positions to points
                for field in "x0", "x1", "top", "bottom", "doctop":
                    w[field] = round(float(w[field]))
                yield w

    def __call__(
        self, infh: Any, pages: Optional[list[int]] = None
    ) -> Iterable[dict[str, Any]]:
        with pdfplumber.open(infh) as pdf:
            yield from self.extract_words(pdf, pages)
