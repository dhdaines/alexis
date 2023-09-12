"""Convertir PDF en XML avec CRF ou règles"""

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import sklearn_crfsuite as crfsuite  # type: ignore
from alexi.crf import CRF, load, page2features, page2labels, split_pages
from sklearn_crfsuite import metrics


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", help="Fichier modele")
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV de test", type=Path)
    return parser


def test(
    crf: crfsuite.CRF,
    test_set: Iterable[dict],
    features="literal",
    labels="simplify",
    n=1,
):
    test = list(test_set)
    X_test = [page2features(p, features, n) for p in test]
    y_test = [page2labels(p, labels) for p in test]
    labels = set(c for c in itertools.chain.from_iterable(y_test) if c.startswith("B-"))
    labels.add("O")
    y_pred = crf.predict(X_test)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels)
    print(report)


def main():
    parser = make_argparse()
    args = parser.parse_args()
    crf = CRF(model=args.model)
    test_set = split_pages(load(args.csvs))
    test(crf.crf, test_set, crf.features, crf.labels, crf.n)


if __name__ == "__main__":
    main()
