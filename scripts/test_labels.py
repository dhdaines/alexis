"""
Use rule-based features and evaluate with CRF evaluator
"""

import argparse
from pathlib import Path
from typing import Iterable

from alexi.crf import load, page2labels
from alexi.label import Classificateur
from alexi.segment import Segmenteur
from sklearn_crfsuite import metrics


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV de test", type=Path)
    return parser


def test(
    test_set: Iterable[dict],
    labels="simplify",
):
    test = list(test_set)
    truth = page2labels(test, labels)
    pred = page2labels(Classificateur()(Segmenteur()(test)), labels)
    labels = set(c for c in truth if c.startswith("B-"))
    labels.add("O")
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report([truth], [pred], labels=sorted_labels)
    print(report)


def main():
    parser = make_argparse()
    args = parser.parse_args()
    test(load(args.csvs))


if __name__ == "__main__":
    main()
