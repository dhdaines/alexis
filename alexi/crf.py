"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
from pathlib import Path
from typing import Iterable, Iterator, Union

import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics


def sign(x: Union[int | float]):
    """Get the sign of a number (should exist...)"""
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def word2features(page, i):
    features = ["bias"]
    features.extend(f"{key}={value}" for key, value in page[i].items() if key != "tag")
    for n in range(1, 3):
        if i > n:
            features.extend(
                f"-{n}:{key}={value}"
                for key, value in page[i - n].items()
                if key != "tag"
            )
        if i < len(page) - n:
            features.extend(
                f"+{n}:{key}={value}"
                for key, value in page[i + n].items()
                if key != "tag"
            )
    return features


def page2features(page):
    return [word2features(page, i) for i in range(len(page))]


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


def page2labels(page):
    def simplify(tag):
        bio, sep, name = tag.partition("-")
        if not name:
            return tag
        return "-".join((bio, TAGMAP.get(name, name)))

    return [simplify(x["tag"]) for x in page]


def page2tokens(page):
    return [x["text"] for x in page]


def split_pages(words: Iterable[dict]) -> list[dict]:
    return [list(p) for idx, p in itertools.groupby(words, operator.itemgetter("page"))]


def load(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        with open(p, "rt") as infh:
            reader = csv.DictReader(infh)
            yield from reader


def train(train_set: Iterable[dict]) -> crfsuite.CRF:
    train_pages = split_pages(train_set)
    nt = len(train_pages) // 10
    X_train = [page2features(s) for s in train_pages[:-nt]]
    y_train = [page2labels(s) for s in train_pages[:-nt]]
    X_dev = [page2features(s) for s in train_pages[-nt:]]
    y_dev = [page2labels(s) for s in train_pages[-nt:]]
    # NOTE: Too much L1 will lead to predicting impossible transitions
    crf = crfsuite.CRF(
        verbose="true",
        algorithm="lbfgs",
        max_iterations=100,
        c1=0.01,
        c2=0.05,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    return crf


def test(crf: crfsuite.CRF, test_set: Iterable[dict]):
    test = list(test_set)
    X_test = [page2features(test)]
    y_test = [page2labels(test)]
    labels = [c for c in crf.classes_ if c.startswith("B-")]
    y_pred = crf.predict(X_test)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels))


if __name__ == "__main__":
    crf = train(load(Path("data/train").glob("*.csv")))
    test(crf, load(Path("data/test").glob("*.csv")))
