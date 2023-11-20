#!/usr/bin/env python3

"""
Extraire les pages pour entrainement du modèle de titre
"""

import argparse
import csv
import itertools
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indir", help="Repertoire d'entrée", type=Path)
    parser.add_argument("outdir", help="Repertoire de sortie", type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for p in args.indir.glob("*.csv"):
        with open(p, "rt") as infh, open(args.outdir / p.name, "wt") as outfh:
            reader = csv.DictReader(infh)
            fieldnames = list(reader.fieldnames)
            fieldnames.insert(0, "seqtag")
            writer = csv.DictWriter(outfh, fieldnames=fieldnames)
            writer.writeheader()
            last_page = None
            for page, group in itertools.groupby(reader, lambda w: w["page"]):
                contents = list(group)
                for w in contents:
                    w["seqtag"] = "O"
                if last_page is None:
                    writer.writerows(contents)
                last_page = contents
            writer.writerows(last_page)


if __name__ == "__main__":
    main()
