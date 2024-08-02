"""Segmentation des textes avec CRF"""

import csv
import itertools
import json
import operator
import re
from collections import Counter
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, Union

import joblib  # type: ignore
import torch
from allennlp_light.modules.conditional_random_field import (
    ConditionalRandomFieldWeightTrans,
)
from tokenizers import Tokenizer  # type: ignore
from torch import nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from alexi.convert import FIELDNAMES
from alexi.format import line_breaks
from alexi.types import T_obj

FEATNAMES = [name for name in FIELDNAMES if name not in ("segment", "sequence")]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"
DEFAULT_MODEL_NOSTRUCT = Path(__file__).parent / "models" / "crf.vl.joblib.gz"
DEFAULT_RNN_MODEL = Path(__file__).parent / "models" / "rnn.pt"
FeatureFunc = Callable[[Sequence[T_obj]], Iterator[list[str]]]


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


def structure_features(page: Iterable[T_obj]) -> Iterator[list[str]]:
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


def layout_features(page: Iterable[T_obj]) -> Iterator[list[str]]:
    """Traits de mise en page pour entrainement d'un modèle."""
    # Split page into lines
    lines = list(line_breaks(list(page)))
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


def literal(page: Iterable[T_obj]) -> Iterator[list[str]]:
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


TITRES = {"Article", "Chapitre", "Section", "SousSection", "Titre"}


def tonly(_, word):
    tag = word.get("segment", "O")
    bio, sep, name = tag.partition("-")
    if not name:
        return tag
    if name in TITRES:
        return tag
    return "O"


def iobonly(_, word):
    tag = word.get("segment", "O")
    bio, _sep, _name = tag.partition("-")
    return bio


LabelFunc = Callable[[int, dict[str, Any]], str]
LABELS: dict[str, LabelFunc] = {
    "literal": lambda _, x: x.get("segment", "O"),
    "bonly": bonly,
    "tonly": tonly,
    "iobonly": iobonly,
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


def retokenize(
    words: Iterable[T_obj], tokenizer: "Tokenizer", drop: bool = False
) -> Iterator[T_obj]:
    """Refaire la tokenisation en alignant les traits et etiquettes."""
    for widx, w in enumerate(words):
        e = tokenizer.encode(w["text"], add_special_tokens=False)
        for tidx, (tok, tid) in enumerate(zip(e.tokens, e.ids)):
            wt = w.copy()
            wt["token"] = tok
            wt["word_id"] = widx
            wt["token_id"] = tid
            if tidx > 0:
                if drop:
                    continue
                for ltype in "sequence", "segment":
                    if ltype in w:
                        label = w[ltype]
                        if label and label[0] == "B":
                            wt[ltype] = f"I-{label[2:]}"
            yield wt


def load(paths: Iterable[PathLike]) -> Iterator[T_obj]:
    for p in paths:
        with open(Path(p), "rt") as infh:
            reader = csv.DictReader(infh)
            for row in reader:
                row["path"] = str(p)
                yield row


def make_fontname(fontname):
    a, plus, b = fontname.partition("+")
    if plus:
        return b
    return fontname


BBOX_FEATS = ["x0", "x1", "top", "bottom"]
DELTA_FEATS = [f"{f}:delta" for f in BBOX_FEATS]
DELTA_DELTA_FEATS = [f"{f}:delta" for f in DELTA_FEATS]
LINE_FEATS = ["line:indent", "line:gap", "line:height"]


def add_deltas(page):
    prev = {}
    for w in page:
        for f in BBOX_FEATS:
            delta = int(w[f]) - prev.setdefault(f, int(w[f]))
            w[f"{f}:delta"] = str(round(delta / 10))
            prev[f] = int(w[f])
    prev = {}
    for w in page:
        for f in DELTA_FEATS:
            delta = int(w[f]) - prev.setdefault(f, int(w[f]))
            w[f"{f}:delta"] = str(round(delta / 10))
            prev[f] = int(w[f])


def make_rnn_features(
    page: Sequence[T_obj],
    labels: str = "literal",
) -> tuple[list[T_obj], list[str]]:
    crf_features = list(
        dict((name, val) for name, _, val in (w.partition("=") for w in feats))
        for feats in layout_features(page)
    )
    rnn_features = []
    maxdim = max(float(page[0][x]) for x in ("page_width", "page_height"))
    prevnum = None
    for f, w in zip(crf_features, page):
        elements = w.get("tagstack", "Span").split(";")
        text = w["text"]
        fontname = make_fontname(w["fontname"])
        feats = {
            "lower": text.lower(),
            "token": w.get("token", ""),
            "fontname": fontname,
            "rgb": w.get("rgb", "#000"),
            "mctag": w.get("mctag", "P"),
            "element": elements[-1],
            "first": f["first"],
            "last": f["last"],
            "uppercase": text.isupper(),
            "title": text.istitle(),
            "punc": bool(PUNC.match(text)),
            "endpunc": bool(ENDPUNC.match(text)),
            "multipunc": bool(MULTIPUNC.match(text)),
            "numeric": text.isnumeric(),
            "bold": ("bold" in fontname.lower()),
            "italic": ("italic" in fontname.lower()),
        }
        bullets = {}
        for pattern in Bullet:
            m = pattern.value.match(text)
            # By definition a bullet comes first in the line
            if m and int(f["first"]):
                bullets[pattern.name] = m.group(1)
        feats["bullet"] = len(bullets) > 0
        sequential = 0
        if int(f["first"]):
            if "NUMERIC" in bullets:
                num = int(bullets["NUMERIC"])
                sequential = int(prevnum is None or num - prevnum == 1)
                prevnum = num
            elif "LOWER" in bullets:
                num = ord(bullets["LOWER"]) - ord("a")
                sequential = int(prevnum is None or num - prevnum == 1)
                prevnum = num
            # print(bool(sequential), text)
        feats["sequential"] = sequential
        for name in BBOX_FEATS:
            val = float(w[name]) / maxdim * 100
            feats[name] = str(round(val))
        for name in LINE_FEATS:
            val = float(f[name]) / maxdim * 100
            feats[name] = str(round(val))
        rnn_features.append(feats)
    add_deltas(rnn_features)
    rnn_labels = list(page2labels(page, labels))
    return rnn_features, rnn_labels


FEATNAMES = (
    [
        "lower",
        "fontname",
        "rgb",
        "mctag",
        "element",
    ]
    + BBOX_FEATS
    + DELTA_FEATS
    + DELTA_DELTA_FEATS
    + LINE_FEATS
)

# Note that these are all binary
VECNAMES = [
    "first",
    "last",
    "sequential",
    "uppercase",
    "punc",
    "endpunc",
    "multipunc",
    "numeric",
    "bold",
    "italic",
    "bullet",
]


def make_page_feats(feat2id, page, featdims):
    return [
        (
            [feat2id[name].get(feats[name], 0) for name in featdims],
            [float(feats[name]) for name in VECNAMES],
        )
        for feats in page
    ]


def make_page_labels(label2id, page):
    return [label2id.get(tag, 0) for tag in page]


def make_rnn_data(
    csvs: Iterable[Path],
    word_dim: int = 32,
    feat_dim: int = 8,
    labels: str = "literal",
    tokenizer: Union["Tokenizer", None] = None,
    min_count: int = 5,
    config: any = None,
):
    """Creer le jeu de donnees pour entrainer un modele RNN."""
    words = filter_tab(load(csvs))
    if tokenizer is not None:
        words = retokenize(words, tokenizer, drop=True)
    pages = split_pages(words)
    if config is not None:
        X, y = zip(*(make_rnn_features(p, labels=config["labels"]) for p in pages))
        # Use the *labels* from this dataset
        label_counts = Counter(itertools.chain.from_iterable(y))
        id2label = sorted(label_counts.keys(), reverse=True)
        # Use the *features* from the other dataset
        feat2id = config["feat2id"]
        featdims = config["featdims"]
    else:
        X, y = zip(*(make_rnn_features(p, labels=labels) for p in pages))
        label_counts = Counter(itertools.chain.from_iterable(y))
        id2label = sorted(label_counts.keys(), reverse=True)
        feat2count: dict[str, Counter] = {name: Counter() for name in FEATNAMES}
        if tokenizer is not None:
            # FIXME: should use all tokens
            feat2count["token"] = Counter()
        for feats in itertools.chain.from_iterable(X):
            for name, val in feats.items():
                if name in feat2count:
                    feat2count[name][val] += 1
        if tokenizer is not None:
            del feat2count["lower"]
        feat2id = {}
        for name, counts in feat2count.items():
            ids = feat2id[name] = {"": 0}
            for val, count in counts.most_common():
                if count < min_count:
                    break
                if val not in ids:
                    ids[val] = len(ids)
            # Eliminate features with only one embedding
            if len(ids) == 1:
                del feat2id[name]
        # FIXME: Should go in train_rnn
        featdims = dict(
            (name, word_dim) if name == "lower" else (name, feat_dim)
            for name in feat2id
        )
    label2id = dict((label, idx) for (idx, label) in enumerate(id2label))
    all_data = [
        (
            make_page_feats(feat2id, page, featdims),
            make_page_labels(label2id, tags),
        )
        for page, tags in zip(X, y)
    ]

    return all_data, featdims, feat2id, label_counts, id2label


def load_rnn_data(
    iobs: Iterable[T_obj],
    feat2id,
    id2label,
    featdims,
    labels: str = "literal",
):
    """Creer le jeu de donnees pour tester un modele RNN."""
    label2id = dict((label, idx) for (idx, label) in enumerate(id2label))
    pages = (make_rnn_features(p, labels=labels) for p in split_pages(iobs))
    all_data = [
        (
            make_page_feats(feat2id, page, featdims),
            make_page_labels(label2id, tags),
        )
        for page, tags in pages
    ]
    return all_data


def batch_sort_key(example):
    features, labels = example
    return -len(labels)


def pad_collate_fn(batch):
    batch.sort(key=batch_sort_key)
    # Don't use a list comprehension here so we can better understand
    sequences_features = []
    sequences_vectors = []
    sequences_labels = []
    lengths = []
    for example in batch:
        features, labels = example
        feats, vector = zip(*features)
        assert len(labels) == len(feats)
        assert len(labels) == len(vector)
        sequences_features.append(torch.LongTensor(feats))
        # sequences_vectors.append(torch.FloatTensor(np.array(vector) / vecmax))
        sequences_vectors.append(torch.FloatTensor(vector))
        sequences_labels.append(torch.LongTensor(labels))
        lengths.append(len(labels))
    lengths = torch.LongTensor(lengths)
    padded_sequences_features = pad_sequence(
        sequences_features, batch_first=True, padding_value=0
    )
    pack_padded_sequences_features = pack_padded_sequence(
        padded_sequences_features, lengths.cpu(), batch_first=True
    )
    padded_sequences_vectors = pad_sequence(
        sequences_vectors, batch_first=True, padding_value=0
    )
    pack_padded_sequences_vectors = pack_padded_sequence(
        padded_sequences_vectors, lengths.cpu(), batch_first=True
    )
    padded_sequences_labels = pad_sequence(
        sequences_labels, batch_first=True, padding_value=-100
    )
    mask = torch.ne(padded_sequences_labels, -100)
    return (
        (pack_padded_sequences_features, pack_padded_sequences_vectors, mask),
        padded_sequences_labels,
    )


def pad_collate_fn_predict(batch):
    # Require data to be externally sorted by length for prediction
    # (otherwise we have no idea which output corresponds to which input! WTF Poutyne!)
    # Don't use a list comprehension here so we can better understand
    sequences_features = []
    sequences_vectors = []
    sequences_labels = []
    lengths = []
    for example in batch:
        features, labels = example
        feats, vector = zip(*features)
        assert len(labels) == len(feats)
        assert len(labels) == len(vector)
        sequences_features.append(torch.LongTensor(feats))
        # sequences_vectors.append(torch.FloatTensor(np.array(vector) / vecmax))
        sequences_vectors.append(torch.FloatTensor(vector))
        sequences_labels.append(torch.LongTensor(labels))
        lengths.append(len(labels))
    max_len = max(lengths)
    len_lens = len(lengths)
    lengths = torch.LongTensor(lengths).cpu()
    # ought to be built into torch...
    # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    mask = torch.arange(max_len).expand(len_lens, max_len) < lengths.unsqueeze(1)
    padded_sequences_features = pad_sequence(
        sequences_features, batch_first=True, padding_value=0
    )
    pack_padded_sequences_features = pack_padded_sequence(
        padded_sequences_features, lengths.cpu(), batch_first=True
    )
    padded_sequences_vectors = pad_sequence(
        sequences_vectors, batch_first=True, padding_value=0
    )
    pack_padded_sequences_vectors = pack_padded_sequence(
        padded_sequences_vectors, lengths.cpu(), batch_first=True
    )
    return (pack_padded_sequences_features, pack_padded_sequences_vectors, mask)


class RNN(nn.Module):
    def __init__(
        self,
        featdims,
        feat2id,
        veclen,
        id2label,
        label_weights=None,
        hidden_size=64,
        num_layer=1,
        bidirectional=True,
        **_kwargs,
    ):
        super().__init__()
        self.hidden_state = None
        self.embedding_layers = {}
        self.featdims = featdims
        for name in featdims:
            self.embedding_layers[name] = nn.Embedding(
                len(feat2id[name]),
                featdims[name],
                padding_idx=0,
            )
            self.add_module(f"embedding_{name}", self.embedding_layers[name])
        dimension = sum(featdims.values()) + veclen
        self.lstm_layer = nn.LSTM(
            input_size=dimension,
            hidden_size=hidden_size,
            num_layers=num_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), len(id2label)
        )

    def forward(
        self,
        features: PackedSequence | torch.Tensor,
        vectors: PackedSequence | torch.Tensor,
        _mask: torch.Tensor,
    ):
        inputs: PackedSequence | torch.Tensor
        # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184
        if isinstance(features, PackedSequence):
            # for idx, name in enumerate(self.featdims):
            #     print(
            #         "WTF",
            #         name,
            #         features.data[:, idx].min(),
            #         features.data[:, idx].max(),
            #         features.data[:, idx],
            #     )
            #     _ = self.embedding_layers[name](features.data[:, idx])
            stack = [
                self.embedding_layers[name](features.data[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors.data)
            inputs = torch.hstack(stack)
            inputs = torch.nn.utils.rnn.PackedSequence(inputs, features.batch_sizes)
        else:
            assert len(features.shape) == 2  # FIXME: support batches
            stack = [
                self.embedding_layers[name](features[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors)
            inputs = torch.hstack(stack)
        lstm_out, self.hidden_state = self.lstm_layer(inputs)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # Make it a "batch" on output
        if len(lstm_out.shape) == 2:
            lstm_out = lstm_out.unsqueeze(0)
        tag_space = self.output_layer(lstm_out)
        tag_space = tag_space.transpose(
            -1, 1
        )  # We need to transpose since it's a sequence (but why?!)
        return tag_space


def bio_transitions(id2label):
    """Constrain transitions (this is not actually useful)"""
    labels_with_boundaries = list(id2label)
    labels_with_boundaries.extend(("START", "END"))

    allowed = []
    for from_label_index, from_label in enumerate(labels_with_boundaries):
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in enumerate(labels_with_boundaries):
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if from_tag == "START":
                if to_tag in ("O", "B"):
                    allowed.append((from_label_index, to_label_index))
            elif to_tag == "END":
                if from_tag in ("O", "B", "I"):
                    allowed.append((from_label_index, to_label_index))
            elif any(
                (
                    # Can always transition to O or B-x
                    to_tag in ("O", "B"),
                    # Can only transition to I-x from B-x or I-x or from I to I
                    to_tag == "I"
                    and from_tag in ("B", "I")
                    and from_entity == to_entity,
                    # Can transition to I from B-x
                    to_tag == "I" and from_tag == "B" and to_entity == "",
                )
            ):
                allowed.append((from_label_index, to_label_index))
    return allowed


class RNNCRF(RNN):
    def __init__(
        self,
        featdims,
        feat2id,
        veclen,
        id2label,
        label_weights=None,
        hidden_size=64,
        num_layer=1,
        bidirectional=True,
        constrain=False,
        **_kwargs,
    ):
        super().__init__(
            featdims,
            feat2id,
            veclen,
            id2label,
            label_weights,
            hidden_size,
            num_layer,
            bidirectional,
        )
        self.crf_layer = ConditionalRandomFieldWeightTrans(
            num_tags=len(id2label),
            label_weights=label_weights,
            constraints=bio_transitions(id2label) if constrain else None,
        )

    def forward(
        self,
        features: PackedSequence | torch.Tensor,
        vectors: PackedSequence | torch.Tensor,
        mask: torch.Tensor,
    ):
        tag_space = super().forward(features, vectors, mask)
        logits = tag_space.transpose(-1, 1)  # We need to transpose it back because wtf
        # Make it a "batch" or CRF gets quite irate
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        paths = self.crf_layer.viterbi_tags(logits, mask)
        labels, _scores = zip(*paths)
        return logits, labels, mask


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


class RNNSegmenteur(Segmenteur):
    model: RNN

    def __init__(self, model: PathLike = DEFAULT_RNN_MODEL, device="cpu"):
        model = Path(model)
        self.device = torch.device(device)
        with open(model.with_suffix(".json"), "rt") as infh:
            self.config = json.load(infh)
        if "crf" in model.name:
            self.model = RNNCRF(**self.config)
        else:
            self.model = RNN(**self.config)
        self.model.load_state_dict(torch.load(model, map_location=torch.device("cpu")))
        self.model.eval()
        self.model.to(device)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for p in split_pages(words):
            page, _labels = make_rnn_features(p)
            features = make_page_feats(
                self.config["feat2id"], page, self.model.featdims
            )
            feats, vector = zip(*features)
            batch = (
                torch.LongTensor(feats, device=self.device),
                torch.FloatTensor(vector, device=self.device),
                torch.ones(len(feats), device=self.device),
            )
            out = self.model(*batch)
            if isinstance(out, tuple):  # is a crf
                _, (labelgen,), _ = out
            else:
                # Encore WTF
                labelgen = out.transpose(1, -1).argmax(-1)[0].cpu()
            for label_id, word in zip(labelgen, p):
                word["segment"] = self.config["id2label"][label_id]
                yield word
