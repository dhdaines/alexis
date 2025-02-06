"""Tester un CRF"""

import argparse
import itertools
from collections import Counter
from pathlib import Path

from sklearn_crfsuite import metrics

from alexi.segment import Segmenteur, load, page2features, page2labels, split_pages


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", help="Fichier modele", type=Path)
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV de test", type=Path)
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    test = list(split_pages(load(args.csvs)))
    multi_predictions = []
    for fold in args.model.parent.glob(f"{args.model.stem}_[0-9]*.gz"):
        crf = Segmenteur(model=fold)
        X_test = [page2features(p, crf.features, crf.n) for p in test]
        y_test = [page2labels(p, crf.labels) for p in test]
        labels = set(
            c for c in itertools.chain.from_iterable(y_test) if c.startswith("B-")
        )
        labels.add("O")
        multi_predictions.append(crf.crf.predict(X_test))
    y_pred = []
    for page in zip(*multi_predictions):
        assert all(len(y) == len(page[0]) for y in page)
        values = []
        for pred in zip(*page):
            counts = Counter(pred)
            values.append(next(iter(counts.most_common()))[0])
        y_pred.append(values)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, zero_division=0.0
    )
    print(report)


if __name__ == "__main__":
    main()
