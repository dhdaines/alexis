"""
Évaluer la pré-segmentation avec les annotations DocLayNet.
"""

import argparse
import itertools
from pathlib import Path

from sklearn_crfsuite import metrics

from alexi.segment import filter_tab, load, split_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", default="yolo", help="Champ pour evaluation")
    parser.add_argument("refdir", type=Path, help="repertoire de references")
    parser.add_argument("hypdir", type=Path, help="repertoire de predictions ")
    args = parser.parse_args()
    csvs = sorted(args.refdir.glob("*.csv"))
    pred_csvs = [args.hypdir / path.name for path in csvs]
    y_true = [
        [w[args.field] for w in page] for page in split_pages(filter_tab(load(csvs)))
    ]
    y_pred = [
        [w[args.field] for w in page]
        for page in split_pages(filter_tab(load(pred_csvs)))
    ]
    labels = set(c for c in itertools.chain.from_iterable(y_true) if c.startswith("B-"))
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report(
        y_true, y_pred, labels=sorted_labels, zero_division=0.0
    )
    print(report)


if __name__ == "__main__":
    main()
