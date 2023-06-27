#!/usr/bin/env pythone

"""
Entraîne des vecteurs de mot à partir des fichiers CSV.
"""

import fasttext
import argparse
import tempfile
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", help="Fichiers CSV d'entree", nargs="+", type=Path)
args = parser.parse_args()

with tempfile.NamedTemporaryFile("wt") as tempfh:
    for p in args.csv:
        with open(p, "rt") as infh:
            reader = csv.DictReader(infh)
            prevx = 0
            prevy = 0
            line = []
            for row in reader:
                if float(row['x0']) < prevx and float(row['doctop']) - prevy > 20:
                    tempfh.write(" ".join(line))
                    tempfh.write("\n")
                    line = []
                line.append(row["text"])
                prevx = float(row["x0"])
                prevy = float(row["doctop"])
            tempfh.write(" ".join(line))
            tempfh.write("\n")
    model = fasttext.train_unsupervised(tempfh.name, dim=16, maxn=5, epoch=25)
    model.save_model("words.fasttext")
