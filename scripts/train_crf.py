"""Entrainer un CRF pour segmentation/identification"""

import argparse
import csv
import itertools
import logging
import numpy as np
import os
from pathlib import Path
from typing import Iterable, Iterator

import joblib  # type: ignore
import sklearn_crfsuite as crfsuite  # type: ignore
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer

from alexi.segment import load, page2features, page2labels, split_pages

LOGGER = logging.getLogger("train-crf")


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
    parser.add_argument("--seed", default=1381, type=int, help="Graine aléatoire")
    parser.add_argument(
        "--min-count",
        default=10,
        type=int,
        help="Seuil d'évaluation pour chaque classification",
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


def run_cv(args: argparse.Namespace, params: dict, X, y):
    if args.cross_validation_folds == 0:
        args.cross_validation_folds = os.cpu_count()
        LOGGER.debug("Using 1 fold per CPU")
    LOGGER.info("Running cross-validation in %d folds", args.cross_validation_folds)
    counts = {}
    for c in itertools.chain.from_iterable(y):
        if c.startswith("B-"):
            count = counts.setdefault(c, 0)
            counts[c] = count + 1
    labels = []
    for c, n in counts.items():
        if n < args.min_count:
            LOGGER.debug("Label %s count %d (excluded)", c, n)
        else:
            LOGGER.debug("Label %s count %d", c, n)
            labels.append(c)
    labels.sort()
    LOGGER.info("Evaluating on: %s", ",".join(labels))
    crf = crfsuite.CRF(**params)
    scoring = {
        "macro_f1": make_scorer(
            metrics.flat_f1_score, labels=labels, average="macro", zero_division=0.0
        ),
        "micro_f1": make_scorer(
            metrics.flat_f1_score, labels=labels, average="micro", zero_division=0.0
        ),
    }
    for name in labels:
        scoring[name] = make_scorer(
            metrics.flat_f1_score,
            labels=[name],
            average="micro",
            zero_division=0.0,
        )
    scores = cross_validate(
        crf,
        X,
        y,
        cv=KFold(args.cross_validation_folds, shuffle=True, random_state=args.seed),
        scoring=scoring,
        return_estimator=True,
        n_jobs=os.cpu_count(),
    )
    LOGGER.info("Macro F1: %.3f", scores["test_macro_f1"].mean())
    LOGGER.info("Micro F1: %.3f", scores["test_micro_f1"].mean())
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


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
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
    if args.cross_validation_folds == 1:
        crf = crfsuite.CRF(**params, verbose=True)
        crf.fit(X, y)
        if args.outfile:
            joblib.dump((crf, args.n, args.features, args.labels), args.outfile)
    else:
        run_cv(args, params, X, y)


if __name__ == "__main__":
    main()
