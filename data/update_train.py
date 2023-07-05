"""Mettre a jour les tags dans un repertoire."""

import argparse
import csv
from pathlib import Path


def word_eq(r1, r2):
    return (
        r1["text"] == r2["text"]
        and abs(float(r1["x0"]) - float(r2["x0"])) < 1
        and abs(float(r1["top"]) - float(r2["top"])) < 1
    )


def update_tags(annodir: Path):
    for p in annodir.iterdir():
        if p.suffix != ".csv":
            continue
        with open(p, "rt") as oldfh:
            old_rows = list(csv.DictReader(oldfh))
            old_iter = iter(old_rows)
            old_row = next(old_iter)
        with open(Path(p.name), "rt") as newfh:
            reader = csv.DictReader(newfh)
            fieldnames = reader.fieldnames
            new_rows = list(reader)
        out_rows = []
        for row in new_rows:
            # Luckily we only *insert* new rows
            if old_row and word_eq(row, old_row):
                # Carry over old tag (keep tags on unseen rows)
                if row["tag"] == "":
                    row["tag"] = old_row["tag"]
                try:
                    old_row = next(old_iter)
                except StopIteration:
                    old_row = None
            out_rows.append(row)
        assert len(out_rows) == len(new_rows)
        with open(p, "wt") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annodirs", nargs="+", type=Path)
    args = parser.parse_args()
    for d in args.annodirs:
        update_tags(d)


if __name__ == "__main__":
    main()
