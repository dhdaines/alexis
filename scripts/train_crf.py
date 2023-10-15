"""Entrainer un CRF pour segmentation/identification"""

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Optional

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from alexi.segment import load, page2features, page2labels, split_pages


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niter", default=69, type=int, help="Nombre d'iterations d'entrainement"
    )
    parser.add_argument("--features", default="vsl", help="Extracteur de traits")
    parser.add_argument("--labels", default="literal", help="Transformateur de classes")
    parser.add_argument(
        "--train-dev", action="store_true", help="Ajouter dev set au train set"
    )
    parser.add_argument("-n", default=2, type=int, help="Largeur du contexte de traits")
    parser.add_argument("-o", "--outfile", help="Fichier destination pour modele")
    return parser


def train(
    train_set: Iterable[dict],
    dev_set: Optional[Iterable[dict]] = None,
    features="vsl",
    labels="literal",
    n=2,
    niter=69,
) -> crfsuite.CRF:
    train_pages = list(split_pages(train_set))
    X_train = [page2features(s, features, n) for s in train_pages]
    y_train = [page2labels(s, labels) for s in train_pages]

    params = {
        "c1": 0.1,
        "c2": 0.1,
        "algorithm": "lbfgs",
        "max_iterations": niter,
        "all_possible_transitions": True,
    }
    crf = crfsuite.CRF(**params, verbose=True)
    if dev_set is not None:
        dev_pages = list(split_pages(dev_set))
        X_dev = [page2features(s, features, n) for s in dev_pages]
        y_dev = [page2labels(s, labels) for s in dev_pages]
        crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    else:
        crf.fit(X_train, y_train)
    for (src, dest), weight in crf.transition_features_.items():
        src_iob, _, src_tag = src.partition("-")
        dest_iob, _, dest_tag = dest.partition("-")
        if src_iob in ("I", "O") and dest_iob == "I" and src_tag != dest_tag:
            crf.transition_features_[src, dest] = -5
    return crf


def main():
    parser = make_argparse()
    args = parser.parse_args()
    train_set = itertools.chain(
        load(Path("data/train").glob("*.csv")),
        load([Path("test/data/pdf_structure.csv")]),
        load([Path("test/data/pdf_figures.csv")]),
    )
    dev_set = load(Path("data/dev").glob("*.csv"))
    if args.train_dev:
        train_set = itertools.chain(train_set, dev_set)
        dev_set = None
    crf = train(
        train_set,
        dev_set,
        features=args.features,
        labels=args.labels,
        n=args.n,
        niter=args.niter,
    )
    if args.outfile:
        joblib.dump((crf, args.n, args.features, args.labels), args.outfile)


if __name__ == "__main__":
    main()
