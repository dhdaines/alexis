import csv
from pathlib import Path

for p in Path("train").iterdir():
    if p.suffix != ".csv":
        continue
    ps = Path(p.name)
    with open(p, "rt") as infh:
        reader = csv.DictReader(infh)
        tags = [row["tag"] for row in reader]
    with open(ps, "rt") as infh:
        reader = csv.DictReader(infh)
        rows = list(reader)
        assert len(rows) == len(tags)
    with open(p, "wt") as outfh:
        writer = csv.DictWriter(outfh, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row, tag in zip(rows, tags):
            row["tag"] = tag
            writer.writerow(row)

