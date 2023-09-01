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
    word = page[i]["text"]
    top = int(page[i]["top"])
    bottom = int(page[i]["bottom"])
    x0 = int(page[i]["x0"])
    mcid = page[i].get("mcid", -1)
    height = bottom - top

    features = [
        "bias",
        "word.lower=%s" % word.lower(),
        "word.isupper=%s" % word.isupper(),
        "word.istitle=%s" % word.istitle(),
        "word.isdigit=%s" % word.isdigit(),
        "word.height5=%d" % round(height / 5),
        "word.x0100=%d" % round(x0 / 100),
        "word.top100=%d" % round(top / 100),
        "word.mctag=" + page[i].get("mctag", ""),
    ]
    if i > 0:
        word1 = page[i - 1]["text"]
        top1 = int(page[i - 1]["top"])
        bottom1 = int(page[i - 1]["bottom"])
        x11 = int(page[i - 1]["x1"])
        mcid1 = page[i].get("mcid", -1)
        height1 = bottom1 - top1
        ydelta = top - bottom1
        xdelta = x0 - x11
        features.extend(
            [
                "-1:word.lower=%s" % word1.lower(),
                "-1:word.istitle=%s" % word1.istitle(),
                "-1:word.isupper=%s" % word1.isupper(),
                "-1:word.height5=%d" % round(height1 / 5),
                "word.ydelta5=%d" % round(ydelta / 5),
                "word.ydelta100=%d" % round(ydelta / 100),
                "word.xdelta100=%d" % round(xdelta / 100),
                "word.ydelta.sign=%d" % sign(ydelta),
                "word.xdelta.sign=%d" % sign(xdelta),
                "word.newheight=%d" % int(height1 != height),
                "word.newmcid=%d" % int(mcid1 != mcid),
            ]
        )
    if i > 1:
        word2 = page[i - 2]["text"]
        features.extend(
            [
                "-2:word.lower=%s" % word2.lower(),
                "-2:word.istitle=%s" % word2.istitle(),
                "-2:word.isupper=%s" % word2.isupper(),
            ]
        )
    if i < len(page) - 1:
        word1 = page[i + 1]["text"]
        features.extend(
            [
                "+1:word.lower=%s" % word1.lower(),
                "+1:word.istitle=%s" % word1.istitle(),
                "+1:word.isupper=%s" % word1.isupper(),
            ]
        )
    if i < len(page) - 2:
        word2 = page[i + 2]["text"]
        features.extend(
            [
                "+2:word.lower=%s" % word2.lower(),
                "+2:word.istitle=%s" % word2.istitle(),
                "+2:word.isupper=%s" % word2.isupper(),
            ]
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
