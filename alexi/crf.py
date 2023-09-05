"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
from pathlib import Path
from typing import Callable, Iterable, Iterator, Union

import eli5
import mlflow
import sklearn_crfsuite as crfsuite
from mlflow import log_metric, log_param, log_params, log_text
from sklearn_crfsuite import metrics

from alexi.convert import FIELDNAMES
from alexi.label import Bullet

FEATNAMES = [name for name in FIELDNAMES if name != "tag"]


def sign(x: Union[int | float]):
    """Get the sign of a number (should exist...)"""
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def make_delta() -> Callable[[int, str], list[str]]:
    prev_word = None

    def delta_one(idx, word):
        nonlocal prev_word
        if idx == 0:
            prev_word = None
        features = pruned(idx, word)
        height = float(word["bottom"]) - float(word["top"])
        if prev_word is None:
            dx = 0
            dy = 0
            dys = 1
        else:
            dx = float(word["x0"]) - float(prev_word["x0"])
            dy = float(word["top"]) - float(prev_word["top"])
            line_gap = float(word["top"]) - float(prev_word["bottom"])
            prev_height = float(prev_word["bottom"]) - float(prev_word["top"])
            dys = line_gap / prev_height
        features.extend(
            [
                "xdelta:" + str(round(dx / height / 10)),
                "xdsign:" + str(sign(dx)),
                "ydelta:" + str(round(dy / height / 10)),
                "ydsign:" + str(sign(dy)),
                "ldelta:" + str(round(dys)),
            ]
        )
        prev_word = word
        return features

    return delta_one


def quantized(_, word):
    features = pruned(_, word)
    ph = float(word["page_height"])
    pw = float(word["page_width"])
    height = float(word["bottom"]) - float(word["top"])
    features.extend(
        [
            "x0:" + str(round(float(word["x0"]) / pw * 10)),
            "x1:" + str(round(pw - float(word["x1"]) / pw * 10)),
            "top:" + str(round(float(word["top"]) / ph * 10)),
            "bottom:" + str(round(ph - float(word["bottom"]) / ph * 10)),
            "height:" + str(round(height / 10)),
        ]
    )
    return features


def pruned(_, word):
    mcid = word.get("mcid")
    if mcid is None:  # UGH ARG WTF
        mcid = ""
    bullet = ""
    for pattern in Bullet:
        if pattern.value.match(word["text"]):
            bullet = pattern.name
    features = [
        "bias",
        "lower3:" + word["text"][0:3].lower(),
        "mctag:" + word.get("mctag", ""),
        "mcid:" + str(mcid),
        "tableau:" + str(word.get("tableau", "")),
        "bullet:" + bullet,
    ]
    return features


def literal(_, word):
    features = ["bias"]
    for key in FEATNAMES:
        feat = word.get(key)
        if feat is None:
            feat = ""
        features.append("=".join((key, str(feat))))
    return features


FEATURES: dict[str, Callable[[int, dict], list[str]]] = {
    "literal": literal,
    "pruned": pruned,
    "quantized": quantized,
    "delta": make_delta(),
}


def page2features(page, features="literal", n=1):
    log_param("features", features)
    log_param("n", n)
    f = FEATURES.get(features, literal)
    features = [f(i, w) for i, w in enumerate(page)]

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


def bonly(tag):
    bio, sep, name = tag.partition("-")
    if not name:
        return tag
    if bio == "I":
        return "I"
    return "-".join((bio, TAGMAP.get(name, name)))


LABELS: dict[str, Callable[str, str]] = {
    "literal": lambda x: x,
    "simplify": simplify,
    "bonly": bonly,
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
    train_set: Iterable[dict], features="literal", labels="simplify", n=1, niter=100
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
        "max_iterations": niter,
        "all_possible_transitions": True,
    }
    log_params(params)
    # NOTE: Too much L1 will lead to predicting impossible transitions
    crf = crfsuite.CRF(verbose="true", **params)
    crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    return crf


HTML_HEAD = """
<html lang="en">
<head>
<meta charset="UTF-8">
</head>
<body>
"""
HTML_FOOT = """
</body>
</html>
"""


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
    explain = eli5.format_as_html(eli5.explain_weights(crf))
    log_text("\n".join((HTML_HEAD, explain, HTML_FOOT)), "explain.html")
    print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="delta", help="Extracteur de traits")
    parser.add_argument(
        "--labels", default="simplify", help="Transformateur de classes"
    )
    parser.add_argument("-n", default=2, type=int, help="Largeur du contexte de traits")
    args = parser.parse_args()
    train_set = load(Path("data/train").glob("*.csv"))
    test_set = load(Path("data/test").glob("*.csv"))
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # OMG WTF
    with mlflow.start_run():
        crf = train(train_set, features=args.features, labels=args.labels, n=args.n)
        test(crf, test_set, features=args.features, labels=args.labels, n=args.n)
