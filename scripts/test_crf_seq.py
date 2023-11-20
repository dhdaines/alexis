"""Tester un CRF"""

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from sklearn_crfsuite import metrics

from alexi.label import load, make_data


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", help="Fichier modele")
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV de test", type=Path)
    return parser


def test(
    crf: crfsuite.CRF,
    test_set: Iterable[dict],
    n=2,
):
    X_test, y_test = make_data(test_set)
    label_names = set(
        c for c in itertools.chain.from_iterable(y_test) if c.startswith("B-")
    )
    y_pred = crf.predict(X_test)
    sorted_label_names = sorted(label_names, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_label_names
    )
    print(report)


def main():
    parser = make_argparse()
    args = parser.parse_args()
    crf = joblib.load(args.model)
    test_set = load(args.csvs)
    test(crf, test_set)


if __name__ == "__main__":
    main()
