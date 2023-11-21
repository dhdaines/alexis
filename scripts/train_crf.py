"""Entrainer un CRF pour segmentation/identification"""

import argparse
import csv
import itertools
import numpy as np
from pathlib import Path
from typing import Iterable, Iterator

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from alexi.segment import load, page2features, page2labels, split_pages


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs", nargs="+", help="Fichiers CSV d'entrainement", type=Path
    )
    parser.add_argument(
        "--niter", default=200, type=int, help="Nombre d'iterations d'entrainement"
    )
    parser.add_argument("--features", default="vsl", help="Extracteur de traits")
    parser.add_argument("--labels", default="literal", help="Transformateur de classes")
    parser.add_argument(
        "--train-dev", action="store_true", help="Ajouter dev set au train set"
    )
    parser.add_argument("-n", default=2, type=int, help="Largeur du contexte de traits")
    parser.add_argument(
        "--c1", default=0.5, type=float, help="Coefficient de regularisation L1"
    )
    parser.add_argument(
        "--c2", default=0.1, type=float, help="Coefficient de regularisation L2"
    )
    parser.add_argument(
        "-x",
        "--cross-validation-folds",
        default=1,
        type=int,
        help="Faire la validation croisée pour évaluer le modèle.",
    )
    parser.add_argument("-o", "--outfile", help="Fichier destination pour modele")
    parser.add_argument("-s", "--scores", help="Fichier destination pour évaluations")
    return parser


def filter_tab(words: Iterable[dict]) -> Iterator[dict]:
    """Enlever les mots dans des tableaux car on va s'en occuper autrement."""
    for w in words:
        if "Tableau" in w["segment"]:
            continue
        if "Table" in w["tagstack"]:
            continue
        yield w


def main():
    parser = make_argparse()
    args = parser.parse_args()
    data = load(args.csvs)
    pages = list(split_pages(filter_tab(data)))
    X = [page2features(s, args.features, args.n) for s in pages]
    y = [page2labels(s, args.labels) for s in pages]
    params = {
        "c1": args.c1,
        "c2": args.c2,
        "algorithm": "lbfgs",
        "max_iterations": args.niter,
        "all_possible_transitions": True,
    }
    crf = crfsuite.CRF(**params, verbose=True)
    labels = list(
        set(c for c in itertools.chain.from_iterable(y) if c.startswith("B-"))
    )
    labels.sort()
    if args.cross_validation_folds == 1:
        crf.fit(X, y)
        if args.outfile:
            joblib.dump((crf, args.n, args.features, args.labels), args.outfile)
    else:
        scoring = {
            "macro_f1": make_scorer(
                metrics.flat_f1_score, labels=labels, average="macro", zero_division=0.0
            ),
        }
        for name in labels:
            scoring[name] = make_scorer(
                metrics.flat_f1_score,
                labels=[name],
                average="macro",
                zero_division=0.0,
            )
        scores = cross_validate(
            crf,
            X,
            y,
            cv=args.cross_validation_folds,
            scoring=scoring,
            return_estimator=True,
        )
        for key, val in scores.items():
            print(f"{key}: {val}")
        if args.outfile:
            for idx, xcrf in enumerate(scores["estimator"]):
                joblib.dump(
                    (xcrf, args.n, args.features, args.labels),
                    args.outfile + f"_{idx + 1}.gz",
                )
        if args.scores:
            with open(args.scores, "wt") as outfh:
                fieldnames = [
                    "Label",
                    "Average",
                    *range(1, args.cross_validation_folds + 1),
                ]
                writer = csv.DictWriter(outfh, fieldnames=fieldnames)
                writer.writeheader()

                def makerow(name, scores):
                    row = {"Label": name, "Average": np.mean(scores)}
                    for idx, score in enumerate(scores):
                        row[idx + 1] = score
                    return row

                writer.writerow(makerow("ALL", scores["test_macro_f1"]))
                for name in labels:
                    writer.writerow(makerow(name, scores[f"test_{name}"]))


if __name__ == "__main__":
    main()
