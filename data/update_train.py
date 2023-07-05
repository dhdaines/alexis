import csv
from pathlib import Path


def word_eq(r1, r2):
    return (r1["text"] == r2["text"]
            and abs(float(r1["x0"]) - float(r2["x0"])) < 0.1
            and abs(float(r1["top"]) - float(r2["top"])) < 0.1)


for p in Path("train").iterdir():
    if p.suffix != ".csv":
        continue
    with open(p, "rt") as infh:
        reader = csv.DictReader(infh)
        old_rows = list(reader)
    ps = Path(p.name)
    with open(ps, "rt") as infh:
        reader = csv.DictReader(infh)
        fieldnames = reader.fieldnames
        new_rows = list(reader)
    out_rows = []
    old_iter = iter(old_rows)
    old_row = next(old_iter)
    for row in new_rows:
        # Luckily we only *insert* new rows
        if old_row and word_eq(row, old_row):
            # Carry over old tag (keep tags on unseen rows)
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
