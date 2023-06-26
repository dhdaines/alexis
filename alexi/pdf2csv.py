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
    return parser


def crop_page(p, margin=72):
    return p.crop((0, margin, p.width, p.height - margin))


def write_csv(pdf, path):
    with open(path, "wt") as ofh:
        fieldnames = ["tag"] + list(pdf.pages[0].extract_words()[0].keys())
        writer = csv.DictWriter(ofh, fieldnames=fieldnames)
        writer.writeheader()
        for p in tqdm(pdf.pages):
            for w in crop_page(p).extract_words():
                writer.writerow(w)


def main(args):
    with pdfplumber.open(args.infile) as pdf:
        write_csv(pdf, args.outfile)


if __name__ == "__main__":
    main(make_argparse().parse_args())

    
