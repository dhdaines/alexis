"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Union

import joblib  # type: ignore

from alexi.convert import FIELDNAMES

FEATNAMES = [name for name in FIELDNAMES if name not in ("segment", "sequence")]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"
FeatureFunc = Callable[[int, dict], list[str]]


class Bullet(Enum):
    NUMERIC = re.compile(r"^(\d+)[\)\.°]$")
    LOWER = re.compile(r"^([a-z])[\)\.]$")
    UPPER = re.compile(r"^([A-Z])[\)\.]$")
    ROMAN = re.compile(r"^([xiv]+)[\)\.]$", re.IGNORECASE)
    BULLET = re.compile(r"^([•-])$")  # FIXME: need more bullets


def sign(x: Union[int | float]):
    """Get the sign of a number (should exist...)"""
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def make_visual_structural_literal() -> FeatureFunc:
    prev_word = None
    prev_line_height = None
    prev_line_start = None

    def visual_one(idx, word):
        nonlocal prev_word, prev_line_height, prev_line_start
        if idx == 0:  # page break
            prev_word = None
            prev_line_start = float(word["x0"])
            prev_line_height = 1  # arbitrary
        ph = float(word["page_height"])
        pw = float(word["page_width"])
        height = float(word["bottom"]) - float(word["top"])
        bullet = None
        for pattern in Bullet:
            if pattern.value.match(word["text"]):
                bullet = pattern.name
                break
        features = [
            "bias",
            "lower:" + word["text"].lower(),
            "x0:%.1f" % (float(word["x0"]) / pw),
            "x1:%.1f" % ((pw - float(word["x1"])) / pw),
            "top:%.1f" % (float(word["top"]) / ph),
            "bottom:%.1f" % ((ph - float(word["bottom"])) / ph),
            "height:%.1f" % (height / 10),
            "bold:%s" % str("bold" in word["fontname"].lower()),
            "italic:%s" % str("italic" in word["fontname"].lower()),
            "bullet:%s" % bullet,
        ]
        newline = False
        linedelta = 0.0
        dx = 1
        dy = 0
        dh = 0
        prev_height = 1
        if prev_word is not None:
            height = float(word["bottom"]) - float(word["top"])
            prev_height = float(prev_word["bottom"]) - float(prev_word["top"])
            dx = float(word["x0"]) - float(prev_word["x0"])
            dy = float(word["top"]) - float(prev_word["top"])
            dh = height - prev_height
            if dx < 0 and dy >= prev_height:
                prev_line_height = prev_height
                newline = True
                linedelta = float(word["x0"]) - prev_line_start
                prev_line_start = float(word["x0"])
        yhdelta = dy / prev_line_height
        features.extend(
            [
                "xdsign:" + str(sign(dx)),
                "ydsign:" + str(sign(dy)),
                "hdsign:" + str(sign(dh)),
                "xdelta:%.1f" % (dx / pw),
                "ydelta:%.1f" % (dy / ph),
                "hdelta:%.1f" % (dh / prev_height),
                "newline:%s" % str(newline),
                "linedelta:%.1f" % (linedelta / pw),
                "yhdelta:%d" % round(min(yhdelta, 5.0)),
            ]
        )
        prev_mcid = prev_word["mcid"] if prev_word is not None else ""
        elements = set(word.get("tagstack", "").split(";"))
        if word["mcid"] != prev_mcid:
            features.append("newmcid")
        mctag = word.get("mctag")
        if mctag:
            features.append("mctag:" + mctag)
        if "Table" in elements:
            features.append("table")
        if "Figure" in elements:
            features.append("figure")
        if "TOCI" in elements:
            features.append("toc")
        prev_word = word
        return features

    return visual_one


def make_visual_literal() -> FeatureFunc:
    prev_word = None
    prev_line_height = None
    prev_line_start = None

    def visual_one(idx, word):
        nonlocal prev_word, prev_line_height, prev_line_start
        if idx == 0:  # page break
            prev_word = None
            prev_line_start = float(word["x0"])
            prev_line_height = 1  # arbitrary
        ph = float(word["page_height"])
        pw = float(word["page_width"])
        height = float(word["bottom"]) - float(word["top"])
        bullet = None
        for pattern in Bullet:
            if pattern.value.match(word["text"]):
                bullet = pattern.name
                break
        features = [
            "bias",
            "lower:" + word["text"].lower(),
            "x0:%.1f" % (float(word["x0"]) / pw),
            "x1:%.1f" % ((pw - float(word["x1"])) / pw),
            "top:%.1f" % (float(word["top"]) / ph),
            "bottom:%.1f" % ((ph - float(word["bottom"])) / ph),
            "height:%.1f" % (height / 10),
            "bold:%s" % str("bold" in word["fontname"].lower()),
            "italic:%s" % str("italic" in word["fontname"].lower()),
            "bullet:%s" % bullet,
            "page_number:%d" % int(word["page"]),
        ]
        newline = False
        linedelta = 0.0
        dx = 1
        dy = 0
        dh = 0
        prev_height = 1
        if prev_word is not None:
            height = float(word["bottom"]) - float(word["top"])
            prev_height = float(prev_word["bottom"]) - float(prev_word["top"])
            dx = float(word["x0"]) - float(prev_word["x0"])
            dy = float(word["top"]) - float(prev_word["top"])
            dh = height - prev_height
            if dx < 0 and dy >= prev_height:
                prev_line_height = prev_height
                newline = True
                linedelta = float(word["x0"]) - prev_line_start
                prev_line_start = float(word["x0"])
        yhdelta = dy / prev_line_height
        features.extend(
            [
                "xdsign:" + str(sign(dx)),
                "ydsign:" + str(sign(dy)),
                "hdsign:" + str(sign(dh)),
                "xdelta:%.1f" % (dx / pw),
                "ydelta:%.1f" % (dy / ph),
                "hdelta:%.1f" % (dh / prev_height),
                "newline:%s" % str(newline),
                "linedelta:%.1f" % (linedelta / pw),
                "yhdelta:%d" % round(min(yhdelta, 5.0)),
            ]
        )
        prev_word = word
        return features

    return visual_one


def make_delta() -> FeatureFunc:
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


FEATURES: dict[str, FeatureFunc] = {
    "literal": literal,
    "pruned": pruned,
    "quantized": quantized,
    "delta": make_delta(),
    "vsl": make_visual_structural_literal(),
    "vl": make_visual_literal(),
}


def page2features(page, feature_func: Union[str, FeatureFunc] = literal, n: int = 1):
    if isinstance(feature_func, str):
        feature_func = FEATURES.get(feature_func, literal)
    features = [feature_func(i, w) for i, w in enumerate(page)]

    def adjacent(features, label):
        return (":".join((label, feature)) for feature in features if feature != "bias")

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
