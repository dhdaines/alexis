"""
Générer des PDF et CSV annotés pour corriger le modèle.
"""

import argparse
import csv
import itertools
import logging
from operator import attrgetter
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from alexi.analyse import group_iob
from alexi.convert import Converteur, write_csv
from alexi.label import Identificateur, DEFAULT_MODEL as DEFAULT_LABEL_MODEL
from alexi.segment import (
    Segmenteur,
    DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL,
    DEFAULT_MODEL_NOSTRUCT,
)

LOGGER = logging.getLogger(Path(__file__).stem)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments to the argparse"""
    parser.add_argument(
        "--segment-model",
        help="Modele CRF",
        type=Path,
    )
    parser.add_argument(
        "--label-model", help="Modele CRF", type=Path, default=DEFAULT_LABEL_MODEL
    )
    parser.add_argument(
        "--pages", help="Liste de numéros de page à extraire, séparés par virgule"
    )
    parser.add_argument(
        "--csv", help="Fichier CSV corriger pour mettre à jour la visualisation"
    )
    parser.add_argument("doc", help="Document en PDF", type=Path)
    parser.add_argument("out", help="Nom de base des fichiers de sortie", type=Path)
    return parser


def annotate_pdf(
    path: Path, pages: list[int], iob: list[dict[str, Any]], outpath: Path
) -> None:
    """
    Marquer les blocs de texte extraits par ALEXI dans un PDF.
    """
    pdf = pdfium.PdfDocument(path)
    inpage = 0
    outpage = 0
    if pages:
        for pagenum in pages:
            # Delete up to the current page
            idx = pagenum - 1
            while inpage < idx:
                pdf.del_page(outpage)
                inpage += 1
            # Don't delete the current page :)
            inpage += 1
            outpage += 1
        while len(pdf) > len(pages):
            pdf.del_page(outpage)
    blocs = group_iob(iob)
    for page, (page_number, group) in zip(
        pdf, itertools.groupby(blocs, attrgetter("page_number"))
    ):
        page_height = page.get_height()
        LOGGER.info("page %d", page_number)
        for bloc in group:
            x0, top, x1, bottom = bloc.bbox
            width = x1 - x0
            height = bottom - top
            y = page_height - bottom
            LOGGER.info("bloc %s à %d, %d, %d, %d", bloc.type, x0, y, width, height)
            path = pdfium_c.FPDFPageObj_CreateNewRect(
                x0 - 1, y - 1, width + 2, height + 2
            )
            pdfium_c.FPDFPath_SetDrawMode(path, pdfium_c.FPDF_FILLMODE_NONE, True)
            if bloc.type in ("Chapitre", "Annexe"):  # Rouge
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 255, 0, 0, 255)
            elif bloc.type == "Section":  # Rose foncé
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 255, 50, 50, 255)
            elif bloc.type == "SousSection":  # Rose moins foncé
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 255, 150, 150, 255)
            elif bloc.type == "Article":  # Rose clair
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 255, 200, 200, 255)
            elif bloc.type == "Liste":  # Bleu-vert (pas du tout rose)
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 0, 200, 150, 255)
            elif bloc.type in ("Tete", "Pied"):  # Jaunâtre
                pdfium_c.FPDFPageObj_SetStrokeColor(path, 200, 200, 50, 255)
            # Autrement noir
            pdfium_c.FPDFPageObj_SetStrokeWidth(path, 1)
            pdfium_c.FPDFPage_InsertObject(page, path)
        pdfium_c.FPDFPage_GenerateContent(page)
    pdf.save(outpath)


def main(args: argparse.Namespace) -> None:
    """Ajouter des anotations à un PDF selon l'extraction ALEXI"""
    if args.csv is not None:
        with open(args.csv, "rt", encoding="utf-8-sig") as infh:
            iob = list(csv.DictReader(infh))
        pages = []
    else:
        if args.segment_model is not None:
            crf = Segmenteur(args.segment_model)
            crf_n = crf
        else:
            crf = Segmenteur(DEFAULT_SEGMENT_MODEL)
            crf_n = Segmenteur(DEFAULT_MODEL_NOSTRUCT)
        crf_s = Identificateur(args.label_model)
        conv = Converteur(args.doc)
        pages = [int(x.strip()) for x in args.pages.split(",")]
        pages.sort()
        feats = conv.extract_words(pages)
        if conv.tree is None:
            LOGGER.warning("Structure logique absente: %s", args.doc)
            segs = crf_n(feats)
        else:
            segs = crf(feats)
        iob = list(crf_s(segs))
        with open(args.out.with_suffix(".csv"), "wt") as outfh:
            write_csv(iob, outfh)
    annotate_pdf(args.doc, pages, iob, args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    # Done by top-level alexi if not running this as script
    parser.add_argument(
        "-v", "--verbose", help="Notification plus verbose", action="store_true"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    main(args)
