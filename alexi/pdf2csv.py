#!/usr/bin/env python3

"""
Convertir un PDF en CSV pour traitement automatique
"""

import argparse
import pdfplumber
import csv
from pathlib import Path

from tqdm import tqdm


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", help="Fichier PDF à traiter", type=Path)
    parser.add_argument("outfile", help="Fichier CSV à créer", type=Path)
    parser.add_argument("-m", "--margin", help="Points de marge", type=int, default=60)
    return parser


def crop_page(p, margin=60):
    return p.crop((0, margin, p.width, p.height - margin))


def write_csv(pdf, path, margin):
    fields = []
    for page in pdf.pages:
        words = page.extract_words()
        if words:
            fields = list(words[0].keys())
            break
    if not fields:
        return
    with open(path, "wt") as ofh:
        fieldnames = ["page", "tag"] + fields
        writer = csv.DictWriter(ofh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(tqdm(pdf.pages)):
            for w in crop_page(p, margin).extract_words():
                w["page"] = idx
                writer.writerow(w)


def main(args):
    with pdfplumber.open(args.infile) as pdf:
        write_csv(pdf, args.outfile, args.margin)


if __name__ == "__main__":
    main(make_argparse().parse_args())
