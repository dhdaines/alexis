"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union, Sequence

import joblib  # type: ignore

from alexi.convert import FIELDNAMES
from alexi.types import T_obj

FEATNAMES = [name for name in FIELDNAMES if name not in ("segment", "sequence")]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"
DEFAULT_MODEL_NOSTRUCT = Path(__file__).parent / "models" / "crf.vl.joblib.gz"
FeatureFunc = Callable[[int, dict], list[str]]


class Bullet(Enum):
    NUMERIC = re.compile(r"^(\d+)[\)\.°]$")
    LOWER = re.compile(r"^([a-z])[\)\.]$")
    UPPER = re.compile(r"^([A-Z])[\)\.]$")
    ROMAN = re.compile(r"^([xiv]+)[\)\.]$", re.IGNORECASE)
    BULLET = re.compile(r"^([•-])$")  # FIXME: need more bullets


def sign(x: Union[int | float]) -> int:
    """Get the sign of a number (should exist...)"""
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def layout_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    """Traits de mise en page pour entrainement d'un modèle."""
    # Split page into lines
    for word in page:
        features = []
        yield features


PUNC = re.compile(r"""^[\.,;:!-—'"“”‘’]+$""")
ENDPUNC = re.compile(r""".*[\.,;:!-—'"“”‘’]$""")
MULTIPUNC = re.compile(r"""^[\.,;:!-—'"“”‘’]{4,}$""")


def textplus_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    """Traits textuelles pour entrainement d'un modèle."""
    # Première ligne de la page est très importante (souvent un en-tête)
    firstline = set(
        word["text"].lower()
        for word in (next(itertools.groupby(page, operator.itemgetter("top")))[1])
    )
    for word in page:
        text: str = word["text"]
        fontname = word["fontname"]
        features = [
            "lower=%s" % text.lower(),
            "uppercase=%s" % text.isupper(),
            "title=%s" % text.istitle(),
            "punc=%s" % bool(PUNC.match(text)),
            "endpunc=%s" % bool(ENDPUNC.match(text)),
            "multipunc=%s" % bool(MULTIPUNC.match(text)),
            "numeric=%s" % text.isnumeric(),
            "bold=%s" % ("bold" in fontname.lower()),
            "italic=%s" % ("italic" in fontname.lower()),
            "head:table=%s" % ("table" in firstline),
            "head:chapitre=%s" % ("chapitre" in firstline),
            "head:annexe=%s" % ("annexe" in firstline),
        ]
        for pattern in Bullet:
            if pattern.value.match(word["text"]):
                features.append("bullet=%s" % pattern.name)
        yield features


def text_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    """Traits textuelles pour entrainement d'un modèle."""
    # Première ligne de la page est très importante (souvent un en-tête)
    firstline = set(
        word["text"].lower()
        for word in (next(itertools.groupby(page, operator.itemgetter("top")))[1])
    )
    for word in page:
        text: str = word["text"]
        features = [
            "lower=%s" % text.lower(),
            "uppercase=%s" % text.isupper(),
            "title=%s" % text.istitle(),
            "punc=%s" % bool(PUNC.match(text)),
            "endpunc=%s" % bool(ENDPUNC.match(text)),
            "multipunc=%s" % bool(MULTIPUNC.match(text)),
            "numeric=%s" % text.isnumeric(),
            "head:table=%s" % ("table" in firstline),
            "head:chapitre=%s" % ("chapitre" in firstline),
            "head:annexe=%s" % ("annexe" in firstline),
        ]
        for pattern in Bullet:
            if pattern.value.match(word["text"]):
                features.append("bullet=%s" % pattern.name)
        yield features


def literal(page: Sequence[T_obj]) -> Iterator[list[str]]:
    for word in page:
        features = []
        for key in FEATNAMES:
            feat = word.get(key)
            if feat is None:
                feat = ""
            features.append("=".join((key, str(feat))))
        yield features


FEATURES: dict[str, FeatureFunc] = {
    "literal": literal,
    "text": text_features,
    "textplus": textplus_features,
}


def page2features(
    page: Sequence[T_obj], feature_func: Union[str, FeatureFunc] = literal, n: int = 1
):
    if isinstance(feature_func, str):
        feature_func_func = FEATURES.get(feature_func, literal)
    else:
        feature_func_func = feature_func
    features = list(feature_func_func(page))

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
    return [["bias", *f] for f in ngram_features]


def bonly(_, word):
    tag = word.get("segment", "O")
    bio, sep, name = tag.partition("-")
    if not name:
        return tag
    if bio == "I":
        return "I"
    return "-".join((bio, name))


LabelFunc = Callable[[int, dict[str, Any]], str]
LABELS: dict[str, LabelFunc] = {
    "literal": lambda _, x: x.get("segment", "O"),
    "bonly": bonly,
}


def page2labels(page, label_func: Union[str, LabelFunc] = "literal"):
    if isinstance(label_func, str):
        label_func = LABELS.get(label_func, LABELS["literal"])
    return [label_func(i, x) for i, x in enumerate(page)]


def page2tokens(page):
    return (x["text"] for x in page)


def split_pages(words: Iterable[dict]) -> Iterable[list[dict]]:
    return (list(p) for idx, p in itertools.groupby(words, operator.itemgetter("page")))


def load(paths: Iterable[PathLike]) -> Iterator[dict]:
    for p in paths:
        with open(Path(p), "rt") as infh:
            reader = csv.DictReader(infh)
            yield from reader


class Segmenteur:
    def __init__(self, model=DEFAULT_MODEL):
        self.crf, self.n, self.features, self.labels = joblib.load(model)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        c1, c2 = itertools.tee(words)
        pred = itertools.chain.from_iterable(
            self.crf.predict_single(
                page2features(p, feature_func=self.features, n=self.n)
            )
            for p in split_pages(c1)
        )
        for label, word in zip(pred, c2):
            word["segment"] = label
            yield word
