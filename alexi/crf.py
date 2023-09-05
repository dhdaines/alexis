"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
from pathlib import Path
from typing import Iterable, Iterator, Union

import mlflow
import sklearn_crfsuite as crfsuite
from mlflow import log_metric, log_param, log_params, log_text
from sklearn_crfsuite import metrics

from alexi.convert import FIELDNAMES

FEATNAMES = [name for name in FIELDNAMES if name != "tag"]


def sign(x: Union[int | float]):
    """Get the sign of a number (should exist...)"""
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def literal(word):
    features = ["bias"]
    for key in FEATNAMES:
        feat = word.get(key)
        if feat is None:
            feat = ""
        features.append("=".join((key, str(feat))))
    return features


FEATURES = {
    "literal": literal,
}


def page2features(page, features="literal", n=1):
    log_param("features", features)
    log_param("n", n)
    f = FEATURES.get(features, literal)
    features = [f(w) for w in page]

    def adjacent(features, label):
        return (":".join((label, feature)) for feature in features)

    ngram_features = [iter(f) for f in features]
    for m in range(1, n):
        for idx in range(len(features) - m):
            ngram_features[idx] = itertools.chain(
                ngram_features[idx], adjacent(features[idx + 1], f"+{m}")
            )
        for idx in range(m, len(features)):
            ngram_features[idx] = itertools.chain(
                ngram_features[idx], adjacent(features[idx - 1], f"-{m}")
            )
    return [list(f) for f in ngram_features]


TAGMAP = dict(
    Amendement="Alinea",
    Attendu="Alinea",
    Annexe="Titre",
    Chapitre="Titre",
    Section="Titre",
    SousSection="Titre",
    Figure="Titre",
    Article="Titre",
)


def simplify(tag):
    bio, sep, name = tag.partition("-")
    if not name:
        return tag
    return "-".join((bio, TAGMAP.get(name, name)))


LABELS = {
    "simplify": simplify,
}


def page2labels(page, labels="simplify"):
    log_param("labels", labels)
    t = LABELS.get(labels, lambda x: x)
    return [t(x["tag"]) for x in page]


def page2tokens(page):
    return [x["text"] for x in page]


def split_pages(words: Iterable[dict]) -> list[dict]:
    return [list(p) for idx, p in itertools.groupby(words, operator.itemgetter("page"))]


def load(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        with open(p, "rt") as infh:
            reader = csv.DictReader(infh)
            yield from reader


def train(
    train_set: Iterable[dict], features="literal", labels="simplify", n=1
) -> crfsuite.CRF:
    train_pages = split_pages(train_set)
    nt = len(train_pages) // 10
    X_train = [page2features(s, features, n) for s in train_pages[:-nt]]
    y_train = [page2labels(s, labels) for s in train_pages[:-nt]]
    X_dev = [page2features(s, features, n) for s in train_pages[-nt:]]
    y_dev = [page2labels(s, labels) for s in train_pages[-nt:]]
    params = {
        "c1": 0.01,
        "c2": 0.05,
        "algorithm": "lbfgs",
        "max_iterations": 100,
        "all_possible_transitions": True,
    }
    log_params(params)
    # NOTE: Too much L1 will lead to predicting impossible transitions
    crf = crfsuite.CRF(verbose="true", **params)
    crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    return crf


def test(
    crf: crfsuite.CRF,
    test_set: Iterable[dict],
    features="literal",
    labels="simplify",
    n=1,
):
    test = list(test_set)
    X_test = [page2features(test, features, n)]
    y_test = [page2labels(test, labels)]
    labels = [c for c in crf.classes_ if c.startswith("B-")]
    y_pred = crf.predict(X_test)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    log_metric(
        "Macro F1 B",
        metrics.flat_f1_score(y_test, y_pred, labels=sorted_labels, average="macro"),
    )
    report = metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels)
    log_text(report, "report.txt")
    print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="literal", help="Extracteur de traits")
    parser.add_argument(
        "--labels", default="simplify", help="Transformateur de classes"
    )
    parser.add_argument("-n", default=1, type=int, help="Largeur du contexte de traits")
    args = parser.parse_args()
    train_set = load(Path("data/train").glob("*.csv"))
    test_set = load(Path("data/test").glob("*.csv"))
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # OMG WTF
    with mlflow.start_run():
        crf = train(train_set, features=args.features, labels=args.labels, n=args.n)
        test(crf, test_set, features=args.features, labels=args.labels, n=args.n)
