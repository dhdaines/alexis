"""Reconnaissance d'objets textuels avec modèles de vision.

Ce repertoire regroupe quelques détecteurs de mise en page pour faire
la pré-segmentation des documents.  Cet étape est utilisée pour
séparer les images et tableaux du texte pour un traitement séparé
(pour le moment ceci consiste à les convertir en images, mais les
tableaux seront pris en charge autrement à l'avenir).

Puisque les conditions d'utilisation sont plus restrictives pour
certains modèles dont YOLOv8, cette étape est facultative, et vous
pouvez toujours utiliser la détection par défaut qui utilise la
structure explicit du PDF (mais celle-ci n'est pas toujours présente
ni correcte)."""

import itertools
import logging
import operator
from collections import deque
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Union

import pdfplumber
from pdfplumber.page import Page
from playa.structtree import (
    StructElement as PDFStructElement,
    StructTree as PDFStructTree,
)
from pdfplumber.utils.geometry import T_bbox, objects_to_bbox

from alexi.analyse import Bloc

LOGGER = logging.getLogger(Path(__file__).stem)


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
        return objects_to_bbox(mcid_objs)


class Objets:
    """Classe de base pour les détecteurs d'objects."""

    def __call__(
        self, pdf_path: PathLike, pages: Union[None, Iterable[int]] = None
    ) -> Iterator[Bloc]:
        """Extraire les rectangles correspondant aux objets qui seront
        représentés par des images."""
        pdf_path = Path(pdf_path)
        pdf = pdfplumber.open(pdf_path)
        try:
            # Get the tree for the *entire* document since elements
            # like the TOC may span multiple pages, and we won't find
            # them if we look at the parent tree for other than the
            # page in which the top element appears (this is the way
            # the structure tree implementation in pdfplumber works,
            # which might be a bug)
            tree = PDFStructTree(pdf.doc)
        except KeyError:
            LOGGER.warning("Arborescence structurel absent dans %s", pdf_path)
            return
        if pages is None:
            pages = range(1, len(pdf.pages) + 1)
        pageset = set(pages)

        # tables *might* span multiple pages (in practice, no...) so
        # we have to split them at page breaks, but also, their
        # top-level elements don't have page numbers for this reason.
        # So, we find them in a first traversal, then gather their
        # children in a second one.
        def gather_elements() -> Iterator[PDFStructElement]:
            """Traverser l'arbre structurel en profondeur pour chercher les
            figures et tableaux."""
            d = deque(tree)
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
            if el.mcids:  # FIXME: this should be asserted in playa
                assert el.page_idx is not None
                for mcid in el.mcids:
                    yield el.page_idx + 1, mcid
            d = deque(el.children)
            while d:
                el = d.popleft()
                if el.mcids:  # FIXME: this should be asserted in playa
                    assert el.page_idx is not None
                    for mcid in el.mcids:
                        yield el.page_idx + 1, mcid
                d.extend(el.children)

        def make_bloc(
            el: PDFStructElement, page_number: int, mcids: Iterable[int]
        ) -> Bloc:
            page = pdf.pages[page_number - 1]
            x0, top, x1, bottom = get_element_bbox(page, el, mcids)
            return Bloc(
                type="Tableau" if el.type == "Table" else el.type,
                contenu=[],
                _page_number=int(page_number),
                _bbox=(round(x0), round(top), round(x1), round(bottom)),
            )

        for el in gather_elements():
            # Note: we must sort them as we can't guarantee they come
            # in any particular order
            mcids = list(get_child_mcids(el))
            mcids.sort()
            for page_number, group in itertools.groupby(mcids, operator.itemgetter(0)):
                if page_number in pageset:
                    yield make_bloc(el, page_number, (mcid for _, mcid in group))
