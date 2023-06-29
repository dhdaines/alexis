#!/usr/bin/env python3

"""
Convertir un PDF en CSV pour traitement automatique
"""

import argparse
import pdfplumber
import logging
import csv
from pathlib import Path
from typing import Optional

from tqdm import tqdm


LOGGER = logging.getLogger("pdf2csv")


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", help="Fichier PDF à traiter", type=Path)
    parser.add_argument("outfile", help="Fichier CSV à créer", type=Path)
    parser.add_argument(
        "-m", "--detect-margins", help="Detecter les marges", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    return parser


def remove_margins(page, words) -> tuple[int, Optional[int]]:
    if not words:
        return words
    margin_top = 0
    margin_bottom = page.height
    l1top = words[0]["top"]
    l1bottom = words[0]["bottom"]
    l1size = l1bottom - l1top
    l1 = [word["text"] for word in words if word["top"] == l1top]
    letop = words[-1]["top"]
    lebottom = words[-1]["bottom"]
    lesize = lebottom - letop
    le = [word["text"] for word in words if word["top"] == letop]
    if len(le) == 1 and le[0].isnumeric() and len(le[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en pied trouvé à %f pt", page.page_number, letop
        )
        margin_bottom = letop
    elif lesize < 9 and (page.height - lebottom) < 100:
        LOGGER.info("page %d: pied de page trouvé à %f pt", page.page_number, letop)
        margin_bottom = letop
        # Il existe parfois deux lignes de pied de page
        for w in words[::-1]:
            if w["top"] == letop:
                continue
            wsize = w["bottom"] - w["top"]
            if wsize < 9 and letop - w["bottom"] < 9:
                LOGGER.info(
                    "page %d: deuxième ligne de pied de page trouvé à %f pt",
                    page.page_number,
                    w["top"],
                )
                margin_bottom = w["top"]
                break
    if len(l1) == 1 and l1[0].isnumeric() and len(l1[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en tête trouvé à %f", page.page_number, l1bottom
        )
        margin_top = l1bottom
    elif l1size < 9 and l1top < 100:
        LOGGER.info("page %d: en-tête trouvé a %f pt", page.page_number, l1bottom)
        margin_top = l1bottom
    return [
        word
        for word in words
        if word["top"] >= margin_top and word["bottom"] <= margin_bottom
    ]


def write_csv(pdf, path, margins=False):
    fields = []
    for page in pdf.pages:
        words = page.extract_words()
        if words:
            fields = list(words[0].keys())
            break
    if not fields:
        return
    with open(path, "wt") as ofh:
        fieldnames = ["tag", "page", "page_width", "page_height"] + fields
        writer = csv.DictWriter(ofh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(tqdm(pdf.pages)):
            words = p.extract_words()
            if margins:
                words = remove_margins(p, words)
            for w in words:
                w["page"] = p.page_number
                w["page_height"] = p.height
                w["page_width"] = p.width
                writer.writerow(w)


def main(args):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.verbose:
        global tqdm
        tqdm = lambda x: x  # noqa: E731
    with pdfplumber.open(args.infile) as pdf:
        logging.info("processing %s", args.infile)
        write_csv(pdf, args.outfile, args.detect_margins)


if __name__ == "__main__":
    main(make_argparse().parse_args())
