"""Conversion de PDF en CSV"""

import itertools
import logging

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pdfplumber

LOGGER = logging.getLogger("convert")


def get_tables(page):
    st = page.structure_tree
    d = deque(st)
    while d:
        el = d.popleft()
        if "children" in el:
            d.extend(el["children"])
        if el["type"] == "Table":
            yield el


def get_child_mcids(el):
    d = deque([el])
    while d:
        el = d.popleft()
        if "children" in el:
            d.extend(el["children"])
        if "mcids" in el:
            yield from el["mcids"]


def unify_bbox(objects):
    itor = iter(objects)
    obj = next(itor)
    x0 = obj["x0"]
    top = obj["top"]
    x1 = obj["x1"]
    bottom = obj["bottom"]
    for obj in itor:
        x0 = min(x0, obj["x0"])
        x1 = max(x1, obj["x1"])
        top = min(top, obj["top"])
        bottom = max(bottom, obj["bottom"])
    return (x0, top, x1, bottom)


def bbox_overlap(w, bbox):
    x0, top, x1, bottom = bbox
    return w["x0"] < x1 and w["x1"] > x0 and w["top"] < bottom and w["bottom"] > top


def get_element_bbox(page, el):
    mcids = set(get_child_mcids(el))
    return unify_bbox(
        c for c in itertools.chain(page.chars, page.images) if c.get("mcid") in mcids
    )


def add_margin(bbox, margin):
    x0, top, x1, bottom = bbox
    return (max(0, x0 - margin), max(0, top - margin), x1 + margin, bottom + margin)


class Converteur:
    imgdir: Optional[Path] = None

    def __init__(self, imgdir=None):
        self.imgdir = imgdir

    def extract_words(self, pdf: pdfplumber.PDF) -> Iterator[dict[str, Any]]:
        for p in pdf.pages:
            words = p.extract_words()
            tables = list(get_tables(p))
            tboxes = []
            for idx, t in enumerate(tables):
                tbox = get_element_bbox(p, t)
                tboxes.append(tbox)
                if self.imgdir is not None:
                    img = p.crop(add_margin(tbox, 10)).to_image(antialias=True)
                    img.save(self.imgdir / f"page{p.page_number}-table{idx + 1}.png")

            # Index characters for lookup
            chars = dict(((c["x0"], c["top"]), c) for c in p.chars)
            LOGGER.info("traitement de la page %d", p.page_number)
            prev_table = None
            for w in words:
                # Extract colour from first character (FIXME: assumes RGB space)
                if c := chars.get((w["x0"], w["top"])):
                    # OMG WTF pdfplumber!!!
                    if isinstance(c["stroking_color"], list) or isinstance(
                        c["stroking_color"], tuple
                    ):
                        if len(c["stroking_color"]) == 1:
                            w["r"] = w["g"] = w["b"] = c["stroking_color"][0]
                        elif len(c["stroking_color"]) == 3:
                            w["r"], w["g"], w["b"] = c["stroking_color"]
                        else:
                            LOGGER.warning(
                                "Espace couleur non pris en charge: %s",
                                c["stroking_color"],
                            )
                    else:
                        w["r"] = w["g"] = w["b"] = c["stroking_color"]
                w["page"] = p.page_number
                w["page_height"] = round(float(p.height))
                w["page_width"] = round(float(p.width))
                for table, tbox in zip(tables, tboxes):
                    if bbox_overlap(w, tbox):
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

    def __call__(self, infh: Any) -> Iterable[dict[str, Any]]:
        with pdfplumber.open(infh) as pdf:
            return self.extract_words(pdf)
