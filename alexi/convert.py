"""Conversion de PDF en CSV"""

import itertools
import logging
import operator
from collections import deque
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from pdfplumber import PDF
from pdfplumber.page import Page
from pdfplumber.structure import PDFStructElement, PDFStructTree, StructTreeMissing
from pdfplumber.utils import geometry
from pdfplumber.utils.geometry import T_bbox

from .types import Bloc, T_obj

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


def bbox_overlaps(obox: T_bbox, bbox: T_bbox) -> bool:
    """Déterminer si deux BBox ont une intersection."""
    ox0, otop, ox1, obottom = obox
    x0, top, x1, bottom = bbox
    return ox0 < x1 and ox1 > x0 and otop < bottom and obottom > top


def merge_overlaps(images: Iterable[Bloc]) -> list[Bloc]:
    """Fusionner des blocs qui se touchent en préservant l'ordre"""
    # FIXME: preserving order maybe not necessary :)
    ordered_images = list(enumerate(images))
    ordered_images.sort(key=lambda x: -geometry.calculate_area(x[1].bbox))
    while True:
        nimg = len(ordered_images)
        new_ordered_images = []
        overlapping = {}
        for idx, image in ordered_images:
            for ydx, other in ordered_images:
                if other is image:
                    continue
                if bbox_overlaps(image.bbox, other.bbox):
                    overlapping[ydx] = other
            if overlapping:
                big_box = geometry.merge_bboxes(
                    (image.bbox, *(other.bbox for other in overlapping.values()))
                )
                LOGGER.info(
                    "image %s overlaps %s merged to %s"
                    % (
                        image.bbox,
                        [other.bbox for other in overlapping.values()],
                        big_box,
                    )
                )
                bloc_types = set(
                    bloc.type
                    for bloc in itertools.chain((image,), overlapping.values())
                )
                image_type = "Tableau" if "Tableau" in bloc_types else "Figure"
                new_image = Bloc(
                    type=image_type,
                    contenu=list(
                        itertools.chain(
                            image.contenu,
                            *(other.contenu for other in overlapping.values()),
                        )
                    ),
                    _bbox=big_box,
                    _page_number=image._page_number,
                )
                for oidx, image in ordered_images:
                    if oidx == idx:
                        new_ordered_images.append((idx, new_image))
                    elif oidx in overlapping:
                        pass
                    else:
                        new_ordered_images.append((oidx, image))
                break
        if overlapping:
            ordered_images = new_ordered_images
        if len(ordered_images) == nimg:
            break
    ordered_images.sort()
    return [img for _, img in ordered_images]


def bbox_contains(bbox: T_bbox, ibox: T_bbox) -> bool:
    """Déterminer si une BBox est contenu entièrement par une autre."""
    x0, top, x1, bottom = bbox
    ix0, itop, ix1, ibottom = ibox
    return ix0 >= x0 and ix1 <= x1 and itop >= top and ibottom <= bottom


def get_element_bbox(page: Page, el: PDFStructElement, mcids: Iterable[int]) -> T_bbox:
    """Obtenir le BBox autour d'un élément structurel."""
    bbox = el.attributes.get("BBox", None)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        top = page.height - y1
        bottom = page.height - y0
        return (x0, top, x1, bottom)
    else:
        mcidset = set(mcids)
        mcid_objs = [
            c
            for c in itertools.chain.from_iterable(page.objects.values())
            if c.get("mcid") in mcidset
        ]
        if not mcid_objs:
            return (-1, -1, -1, -1)  # An impossible BBox
        return geometry.objects_to_bbox(mcid_objs)


def get_rgb(c: T_obj) -> str:
    """Extraire la couleur d'un objet en 3 chiffres hexadécimaux"""
    couleur = c.get("non_stroking_color", c.get("stroking_color"))
    if couleur is None:
        return "#000"
    elif len(couleur) == 1:
        r = g = b = couleur[0]
    elif len(couleur) == 3:
        r, g, b = couleur
    else:
        LOGGER.warning("Espace couleur non pris en charge: %s", couleur)
        return "#000"
    return "#" + "".join(("%x" % int(min(0.999, val) * 16) for val in (r, g, b)))


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
    tree: Optional[PDFStructTree]
    y_tolerance: int

    def __init__(
        self,
        path_or_fp: Union[str, Path, BufferedReader, BytesIO],
        y_tolerance: int = 2,
    ):
        self.pdf = PDF.open(path_or_fp)
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
                yield get_word_features(word, page, chars, elmap)

    def make_bloc(
        self, el: PDFStructElement, page_number: int, mcids: Iterable[int]
    ) -> Bloc:
        page = self.pdf.pages[page_number - 1]
        x0, top, x1, bottom = get_element_bbox(page, el, mcids)
        return Bloc(
            type="Tableau" if el.type == "Table" else el.type,
            contenu=[],
            _page_number=int(page_number),
            _bbox=(round(x0), round(top), round(x1), round(bottom)),
        )

    def extract_images(self, pages: Optional[Iterable[int]] = None) -> Iterator[Bloc]:
        """Trouver des éléments qui seront représentés par des images
        (tableaux et figures pour le moment)"""
        if self.tree is None:
            return
        if pages is None:
            pages = range(1, len(self.pdf.pages) + 1)
        pageset = set(pages)

        # tables *might* span multiple pages (in practice, no...) so
        # we have to split them at page breaks, but also, their
        # top-level elements don't have page numbers for this reason.
        # So, we find them in a first traversal, then gather their
        # children in a second one.
        def gather_elements() -> Iterator[PDFStructElement]:
            """Traverser l'arbre structurel en profondeur pour chercher les
            figures et tableaux."""
            if self.tree is None:
                return
            d = deque(self.tree)
            while d:
                el = d.popleft()
                if el.type == "Table":
                    yield el
                elif el.type == "Figure":
                    yield el
                else:
                    d.extendleft(reversed(el.children))

        def get_child_mcids(el: PDFStructElement) -> Iterator[tuple[int, int]]:
            """Trouver tous les MCIDs (avec numeros de page, sinon ils sont
            inutiles!) à l'intérieur d'un élément structurel"""
            for mcid in el.mcids:
                assert el.page_number is not None
                yield el.page_number, mcid
            d = deque(el.children)
            while d:
                el = d.popleft()
                for mcid in el.mcids:
                    assert el.page_number is not None
                    yield el.page_number, mcid
                d.extend(el.children)

        for el in gather_elements():
            # Note: we must sort them as we can't guarantee they come in any particular order
            mcids = list(get_child_mcids(el))
            mcids.sort()
            for page_number, group in itertools.groupby(mcids, operator.itemgetter(0)):
                if page_number in pageset:
                    yield self.make_bloc(el, page_number, (mcid for _, mcid in group))
