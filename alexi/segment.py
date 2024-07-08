"""Segmentation des textes avec CRF"""

import csv
import itertools
import operator
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, Union

import joblib  # type: ignore

from alexi.convert import FIELDNAMES
from alexi.format import line_breaks
from alexi.types import T_obj

FEATNAMES = [name for name in FIELDNAMES if name not in ("segment", "sequence")]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"
DEFAULT_MODEL_NOSTRUCT = Path(__file__).parent / "models" / "crf.vl.joblib.gz"
FeatureFunc = Callable[[Sequence[T_obj]], Iterator[list[str]]]

if False:
    from tokenizers import Tokenizer  # STFU, pyflakes


class Bullet(Enum):
    NUMERIC = re.compile(r"^(\d+)[\)\.°-]$")
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


def structure_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    """Traits de structure logique pour entrainement d'un modèle."""
    for word in page:
        elements = set(word.get("tagstack", "Span").split(";"))
        header = False
        for el in elements:
            if el and el[0] == "H":
                header = True
        features = [
            "toc=%d" % ("TOCI" in elements),
            "mctag=%s" % word.get("mctag", "P"),
            "header=%s" % header,
        ]
        yield features


def layout_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    """Traits de mise en page pour entrainement d'un modèle."""
    # Split page into lines
    lines = list(line_breaks(page))
    prev_line_features: dict[str, int] = {}
    for line in lines:
        page_height = int(line[0]["page_height"])
        line_features = {
            "height": max(int(word["bottom"]) - int(word["top"]) for word in line),
            "left": int(line[0]["x0"]),
            "right": int(line[-1]["x1"]),
            "top": min(int(word["top"]) for word in line),
            "bottom": max(int(word["bottom"]) for word in line),
        }
        for idx in range(len(line)):  # , word in enumerate(line):
            features = [
                "first=%d" % (idx == 0),
                "last=%d" % (idx == len(line) - 1),
                "line:height=%d" % line_features["height"],
                "line:left=%d" % line_features["left"],
                "line:top=%d" % line_features["top"],
                "line:bottom=%d" % (page_height - line_features["bottom"]),
                "line:gap=%d"
                % (line_features["top"] - prev_line_features.get("bottom", 0)),
                "line:indent=%d"
                % (
                    line_features["left"]
                    - prev_line_features.get("left", line_features["left"])
                ),
            ]
            yield features
        prev_line_features = line_features


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
            "rgb=%s" % word.get("rgb", "#000"),
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


def textpluslayout_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    return (tpf + lf for tpf, lf in zip(textplus_features(page), layout_features(page)))


def textpluslayoutplusstructure_features(page: Sequence[T_obj]) -> Iterator[list[str]]:
    return (
        tpf + lf + sf
        for tpf, lf, sf in zip(
            textplus_features(page), layout_features(page), structure_features(page)
        )
    )


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
    "text+": textplus_features,
    "layout": layout_features,
    "text+layout": textpluslayout_features,
    "structure": structure_features,
    "text+layout+structure": textpluslayoutplusstructure_features,
}


def page2features(
    page: Sequence[T_obj], feature_func: Union[str, FeatureFunc] = literal, n: int = 1
):
    if isinstance(feature_func, str):
        feature_func_func = FEATURES[feature_func]
    else:
        feature_func_func = feature_func
    features = list(feature_func_func(page))

    def adjacent(features, label):
        return (
            ":".join((label, feature)) for feature in features if ":" not in feature
        )

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


def split_pages(words: Iterable[T_obj]) -> Iterable[list[T_obj]]:
    return (list(p) for idx, p in itertools.groupby(words, operator.itemgetter("page")))


def filter_tab(words: Iterable[T_obj]) -> Iterator[T_obj]:
    """Enlever les mots dans des tableaux car on va s'en occuper autrement."""
    for w in words:
        if "Tableau" in w["segment"]:
            continue
        if "Table" in w["tagstack"]:
            continue
        yield w


def retokenize(words: Iterable[T_obj], tokenizer: "Tokenizer") -> Iterator[T_obj]:
    """Refaire la tokenisation en alignant les traits et etiquettes.

    Notez que parce que le positionnement de chaque sous-mot sera
    identique aux autres, les modeles de mise en page risquent de ne
    pas bien marcher.  Il serait preferable d'appliquer la
    tokenisation directement sur les caracteres.
    """
    for widx, w in enumerate(words):
        e = tokenizer.encode(w["text"], add_special_tokens=False)
        for tidx, (tok, tid) in enumerate(zip(e.tokens, e.ids)):
            wt = w.copy()
            wt["text"] = tok
            wt["word"] = w["text"]
            wt["word_id"] = widx
            wt["token_id"] = tid
            # Change B to I for subtokens
            if tidx > 0:
                for ltype in "sequence", "segment":
                    if ltype in w:
                        label = w[ltype]
                        if label and label[0] == "B":
                            wt[ltype] = f"I-{label[2:]}"
            yield wt


def detokenize(words: Iterable[T_obj], _tokenizer: "Tokenizer") -> Iterator[T_obj]:
    """Defaire la retokenisation"""
    widx = -1
    for w in words:
        if w["word_id"] != widx:
            widx = w["word_id"]
            w["text"] = w["word"]
            del w["word"]
            del w["word_id"]
            del w["token_id"]
            yield w


def load(paths: Iterable[PathLike]) -> Iterator[T_obj]:
    for p in paths:
        with open(Path(p), "rt") as infh:
            reader = csv.DictReader(infh)
            for row in reader:
                row["path"] = str(p)
                yield row


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
