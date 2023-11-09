"""Entrainer un CRF pour segmentation/identification"""

import argparse
import itertools
from pathlib import Path

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from alexi.label import load, make_data


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niter", default=100, type=int, help="Nombre d'iterations d'entrainement"
    )
    parser.add_argument(
        "--train-dev", action="store_true", help="Ajouter dev set au train set"
    )
    parser.add_argument(
        "--c1", default=0.5, type=float, help="Coefficient de regularisation L1"
    )
    parser.add_argument(
        "--c2", default=0.1, type=float, help="Coefficient de regularisation L2"
    )
    parser.add_argument("-o", "--outfile", help="Fichier destination pour modele")
    return parser


def train(
    train_set,
    dev_set,
    n=2,
    niter=69,
    c1=0.1,
    c2=0.1,
) -> crfsuite.CRF:
    X_train, y_train = make_data(train_set, n)

    params = {
        "c1": c1,
        "c2": c2,
        "algorithm": "lbfgs",
        "max_iterations": niter,
        "all_possible_transitions": True,
    }
    crf = crfsuite.CRF(**params, verbose=True)
    if dev_set is not None:
        X_dev, y_dev = make_data(dev_set, n)
        crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    else:
        crf.fit(X_train, y_train)
    return crf


def main():
    parser = make_argparse()
    args = parser.parse_args()
    train_set = load(Path("data/train").glob("*.csv"))
    dev_set = load(Path("data/dev").glob("*.csv"))
    if args.train_dev:
        train_set = itertools.chain(train_set, dev_set)
        dev_set = None
    crf = train(
        train_set,
        dev_set,
        niter=args.niter,
        c1=args.c1,
        c2=args.c2,
    )
    if args.outfile:
        joblib.dump(crf, args.outfile)


if __name__ == "__main__":
    main()
