#!/usr/bin/env python3

"""
Description des traits importants d'un modèle.
"""
import argparse
from pathlib import Path
import eli5

from alexi.segment import Segmenteur


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Fichier modele", type=Path)
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    crf = Segmenteur(model=args.model)
    print(eli5.format_as_html(eli5.explain_weights(crf.crf)))


if __name__ == "__main__":
    main()
