"""Entrainer un CRF pour segmentation/identification"""

import argparse
from pathlib import Path
from typing import Iterable

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from alexi.crf import load, page2features, page2labels, split_pages


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="vsl", help="Extracteur de traits")
    parser.add_argument(
        "--labels", default="simplify", help="Transformateur de classes"
    )
    parser.add_argument("-n", default=2, type=int, help="Largeur du contexte de traits")
    parser.add_argument("-o", "--outfile", help="Fichier destination pour modele")
    return parser


def train(
    train_set: Iterable[dict], features="literal", labels="simplify", n=1
) -> crfsuite.CRF:
    train_pages = list(split_pages(train_set))
    X_train = [page2features(s, features, n) for s in train_pages]
    y_train = [page2labels(s, labels) for s in train_pages]

    # FIXME: Need a real dev set
    # X_dev = [page2features(s, features, n) for s in train_pages[-nt:]]
    # y_dev = [page2labels(s, labels) for s in train_pages[-nt:]]

    # NOTE: Too much L1 will lead to predicting impossible
    # transitions, can we fix them up in the parameters after the
    # fact?
    params = {
        "c1": 0.01,
        "c2": 0.05,
        "algorithm": "lbfgs",
        "max_iterations": 100,
        "all_possible_transitions": True,
    }
    crf = crfsuite.CRF(**params, verbose=True)
    crf.fit(X_train, y_train)  # , X_dev=X_dev, y_dev=y_dev)
    return crf


def main():
    parser = make_argparse()
    args = parser.parse_args()
    train_set = load(Path("data/train").glob("*.csv"))
    crf = train(train_set, features=args.features, labels=args.labels, n=args.n)
    if args.outfile:
        joblib.dump((crf, args.n, args.features, args.labels), args.outfile)


if __name__ == "__main__":
    main()
