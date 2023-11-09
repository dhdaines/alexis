#!/usr/bin/env python

"""
Convertir les règlements en HTML structuré.
"""

import argparse
import csv
import itertools
import logging
from pathlib import Path
from typing import Any

from alexi.analyse import Analyseur
from alexi.convert import Converteur
from alexi.segment import Segmenteur
from alexi.format import format_html, format_text
from alexi.label import Extracteur

LOGGER = logging.getLogger("convert")


def make_argparse():
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--outdir", help="Repertoire de sortie", type=Path, default="html"
    )
    parser.add_argument(
        "-v", "--verbose", help="Notification plus verbose", action="store_true"
    )
    parser.add_argument(
        "docs", help="Documents en PDF ou CSV pré-annoté", type=Path, nargs="+"
    )
    return parser


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "rt") as infh:
        return list(csv.DictReader(infh))


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    crf = None
    extracteur = Extracteur()
    analyseur = Analyseur()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in args.docs:
        conv = None
        if path.suffix == ".csv":
            LOGGER.info("Lecture de %s", path)
            iob = read_csv(path)
        elif path.suffix == ".pdf":
            LOGGER.info("Conversion, segmentation et classification de %s", path)
            conv = Converteur(path)
            feats = conv.extract_words()
            if crf is None:
                crf = Segmenteur()
            iob = list(extracteur(crf(feats)))

        pdf_path = path.with_suffix(".pdf")
        if conv is None and pdf_path.exists():
            conv = Converteur(pdf_path)

        docdir = args.outdir / path.stem
        LOGGER.info("Génération de pages HTML sous %s", docdir)
        docdir.mkdir(exist_ok=True)

        LOGGER.info("Analyse de la structure de %s", path)
        if conv:
            tables = list(conv.extract_tables())
            figures = list(conv.extract_figures())
            for bloc in itertools.chain(tables, figures):
                img = (
                    conv.pdf.pages[bloc.page_number - 1]
                    .crop(bloc.bbox)
                    .to_image(resolution=150, antialias=True)
                )
                LOGGER.info("Extraction de %s", docdir / bloc.img)
                img.save(docdir / bloc.img)
            doc = analyseur(iob, tables, figures)
        else:
            tables = figures = None
            doc = analyseur(iob)

        for palier, elements in doc.paliers.items():
            for idx, element in enumerate(elements):
                if palier == "Document":
                    element = None
                    title = "index"
                else:
                    title = f"{palier}_{idx+1}"
                with open(docdir / f"{title}.html", "wt") as outfh:
                    LOGGER.info("Génération de %s/%s.html", docdir, title)
                    outfh.write(format_html(doc, element=element))
                with open(docdir / f"{title}.txt", "wt") as outfh:
                    LOGGER.info("Génération de %s/%s.txt", docdir, title)
                    outfh.write(format_text(doc, element=element))


if __name__ == "__main__":
    main()
