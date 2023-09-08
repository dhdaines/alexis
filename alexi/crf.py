"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Union

import joblib

from alexi.convert import FIELDNAMES
from alexi.label import Bullet

FEATNAMES = [name for name in FIELDNAMES if name != "tag"]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"


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
        if idx == 0:  # page break
            prev_word = None
        features = quantized(idx, word)
        ph = float(word["page_height"])
        pw = float(word["page_width"])
        if prev_word is None:
            dx = 1
            dy = 0
            dh = 0
            prev_height = 1
        else:
            height = float(word["bottom"]) - float(word["top"])
            prev_height = float(prev_word["bottom"]) - float(prev_word["top"])
            dx = float(word["x0"]) - float(prev_word["x0"])
            dy = float(word["top"]) - float(prev_word["top"])
            dh = height - prev_height
        features.extend(
            [
                "xdsign:" + str(sign(dx)),
                "ydsign:" + str(sign(dy)),
                "hdsign:" + str(sign(dh)),
                "xdelta:%.1f" % (dx / pw),
                "ydelta:%.1f" % (dy / ph),
                "hdelta:%.1f" % (dh / prev_height),
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
            "x0:%.1f" % (float(word["x0"]) / pw),
            "x1:%.1f" % ((pw - float(word["x1"])) / pw),
            "top:%.1f" % (float(word["top"]) / ph),
            "bottom:%.1f" % ((ph - float(word["bottom"])) / ph),
            "height:%.1f" % (height / 10),
        ]
    )
    return features


def pruned(_, word):
    bullet = ""
    for pattern in Bullet:
        if pattern.value.match(word["text"]):
            bullet = pattern.name
    features = [
        "bias",
        "lower:" + word["text"].lower(),
        "mctag:" + word.get("mctag", ""),
        "tableau:" + str(word.get("tableau") not in ("", None)),
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
    t = LABELS.get(labels, lambda x: x)
    return [t(x["tag"]) for x in page]


def page2tokens(page):
    return (x["text"] for x in page)


def split_pages(words: Iterable[dict]) -> list[dict]:
    return (list(p) for idx, p in itertools.groupby(words, operator.itemgetter("page")))


def load(paths: Iterable[Path]) -> Iterator[dict]:
    for p in paths:
        with open(p, "rt") as infh:
            reader = csv.DictReader(infh)
            yield from reader


class CRF:
    def __init__(self, model=DEFAULT_MODEL):
        self.crf, self.n, self.features, self.labels = joblib.load(model)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        c1, c2 = itertools.tee(words)
        pred = itertools.chain.from_iterable(
            self.crf.predict_single(page2features(p, features=self.features, n=self.n))
            for p in split_pages(c1)
        )
        for label, word in zip(pred, c2):
            word["tag"] = label
            yield word
