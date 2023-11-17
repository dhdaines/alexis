"""Mettre a jour les tags dans un repertoire."""

import argparse
import csv
import logging
from pathlib import Path


def key(row: dict):
    return (row["text"], row["page"], row["x0"], row["top"])


def update_tags(annodir: Path):
    for p in annodir.iterdir():
        if p.suffix != ".csv":
            continue
        if not Path(p.name).exists():
            continue
        old_tags = {}
        with open(p, "rt") as oldfh:
            for row in csv.DictReader(oldfh):
                old_tags[key(row)] = row["tag"]
        out_rows = []
        with open(Path(p.name), "rt") as newfh:
            reader = csv.DictReader(newfh)
            fieldnames = reader.fieldnames
            for row in reader:
                if not row["tag"]:
                    row["tag"] = old_tags.get(key(row), "O")
                out_rows.append(row)
        with open(p, "wt") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annodirs", nargs="+", type=Path)
    args = parser.parse_args()
    logging.basicConfig()
    for d in args.annodirs:
        update_tags(d)


if __name__ == "__main__":
    main()
