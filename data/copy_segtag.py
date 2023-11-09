"""Mettre a jour les tags dans un repertoire."""

import csv
import logging
from pathlib import Path


def key(row: dict):
    return (row["text"], row["page"], row["x0"], row["top"])


def update_tags(annodir: Path, segdir: Path):
    for p in segdir.iterdir():
        if p.suffix != ".csv":
            continue
        basep = annodir / p.name
        if not Path(basep).exists():
            continue
        new_tags = {}
        with open(p, "rt") as newfh:
            for row in csv.DictReader(newfh):
                new_tags[key(row)] = row["seqtag"]
        out_rows = []
        with open(basep, "rt") as basefh:
            reader = csv.DictReader(basefh)
            fieldnames = list(reader.fieldnames)
            for row in reader:
                row["segtag"] = row["tag"]
                del row["tag"]
                row["seqtag"] = new_tags.get(key(row), "O")
                out_rows.append(row)
            fieldnames.remove("tag")
            fieldnames.insert(0, "segtag")
            fieldnames.insert(0, "seqtag")
        with open(basep, "wt") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)


def main():
    logging.basicConfig()
    for d in "train", "dev", "test":
        update_tags(Path(d), Path(f"{d}-title"))


if __name__ == "__main__":
    main()
