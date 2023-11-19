"""Retagger les Tableau/TOC dans les données avec Liste/Alinea/Alouette"""

import argparse
import csv
import itertools
import sys

from alexi.segment import Segmenteur, page2features, split_pages


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", help="Fichier modele")
    parser.add_argument("csv", help="Fichier CSV", type=argparse.FileType("rt"))
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    crf = Segmenteur(model=args.model)
    reader = csv.DictReader(args.csv)
    writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
    words = list(reader)
    X_test = [page2features(p, "vsl", 2) for p in split_pages(words)]
    y_pred = crf.crf.predict(X_test)

    writer.writeheader()
    for label, word in zip(itertools.chain.from_iterable(y_pred), words):
        bio, sep, name = word["segment"].partition("-")
        if name == "Tableau":
            word["segment"] = label
        writer.writerow(word)


if __name__ == "__main__":
    main()
