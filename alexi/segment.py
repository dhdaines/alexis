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
from torch import nn
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from allennlp_light.modules.conditional_random_field.conditional_random_field import (
    allowed_transitions,
)
from allennlp_light.modules.conditional_random_field import (
    ConditionalRandomFieldWeightEmission,
    ConditionalRandomFieldWeightTrans,
    ConditionalRandomFieldWeightLannoy,
)

from alexi.convert import FIELDNAMES
from alexi.format import line_breaks
from alexi.types import T_obj

FEATNAMES = [name for name in FIELDNAMES if name not in ("segment", "sequence")]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crf.joblib.gz"
DEFAULT_MODEL_NOSTRUCT = Path(__file__).parent / "models" / "crf.vl.joblib.gz"
DEFAULT_RNN_MODEL = Path(__file__).parent / "models" / "rnn.pt"
FeatureFunc = Callable[[Sequence[T_obj]], Iterator[list[str]]]

if False:
    from tokenizers import Tokenizer  # type: ignore


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


def make_fontname(fontname):
    a, plus, b = fontname.partition("+")
    if plus:
        return b
    return fontname


def add_deltas(page):
    prev = {}
    for w in page:
        for f in list(f for f in w if f.startswith("v:")):
            w[f"{f}:delta"] = w[f] - prev.setdefault(f, w[f])
            prev[f] = w[f]
    prev = {}
    for w in page:
        for f in list(f for f in w if f.endswith(":delta")):
            w[f"{f}:delta"] = w[f] - prev.setdefault(f, w[f])
            prev[f] = w[f]


def make_rnn_features(
    page: Iterable[T_obj],
    features: str = "text+layout+structure",
    labels: str = "literal",
):
    features = list(
        dict((name, val) for name, _, val in (w.partition("=") for w in feats))
        for feats in page2features(page, features)
    )
    for f, w in zip(features, page):
        f["line:left"] = float(f["line:left"]) / float(w["page_width"])
        f["line:top"] = float(f["line:top"]) / float(w["page_height"])
        f["v:top"] = float(w["top"]) / float(w["page_height"])
        f["v:left"] = float(w["x0"]) / float(w["page_width"])
        f["v:top"] = float(w["top"]) / float(w["page_height"])
        f["v:right"] = (float(w["page_width"]) - float(w["x1"])) / float(
            w["page_width"]
        )
        f["v:bottom"] = (float(w["page_height"]) - float(w["bottom"])) / float(
            w["page_height"]
        )

    add_deltas(features)
    labels = list(page2labels(page, labels))
    return features, labels


FEATNAMES = [
    "lower",
    "rgb",
    "mctag",
    "uppercase",
    "title",
    "punc",
    "endpunc",
    "numeric",
    "bold",
    "italic",
    "toc",
    "header",
    "head:table",
    "head:chapitre",
    "head:annexe",
    "line:height",
    "line:indent",
    "line:gap",
    "first",
    "last",
]

VECNAMES = [
    "line:left",
    "line:top",
    "v:left",
    "v:top",
    "v:right",
    "v:bottom",
    "v:left:delta",
    "v:top:delta",
    "v:right:delta",
    "v:bottom:delta",
    "v:left:delta:delta",
    "v:top:delta:delta",
    "v:right:delta:delta",
    "v:bottom:delta:delta",
]


def make_page_feats(feat2id, page):
    return [
        (
            [feat2id[name].get(feats[name], 0) for name in FEATNAMES],
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
    features: str = "text+layout_structure",
    labels: str = "literal",
):
    """Creer le jeu de donnees pour entrainer un modele RNN."""
    X, y = zip(
        *(
            make_rnn_features(p, features=features, labels=labels)
            for p in split_pages(filter_tab(load(csvs)))
        )
    )
    label_counts = Counter(itertools.chain.from_iterable(y))
    id2label = sorted(label_counts.keys(), reverse=True)
    label2id = dict((label, idx) for (idx, label) in enumerate(id2label))
    feat2id = {name: {"": 0} for name in FEATNAMES}
    for feats in itertools.chain.from_iterable(X):
        for name, ids in feat2id.items():
            if feats[name] not in ids:
                ids[feats[name]] = len(ids)

    all_data = [
        (
            make_page_feats(feat2id, page),
            make_page_labels(label2id, labels),
        )
        for page, labels in zip(X, y)
    ]
    # FIXME: Should go in train_rnn
    featdims = dict(
        (name, word_dim) if name == "lower" else (name, feat_dim) for name in FEATNAMES
    )

    return all_data, featdims, feat2id, label_counts, id2label


def load_rnn_data(iobs: Iterable[T_obj], feat2id, id2label):
    """Creer le jeu de donnees pour entrainer un modele RNN."""
    label2id = dict((label, idx) for (idx, label) in enumerate(id2label))
    pages = (make_rnn_features(p) for p in split_pages(iobs))
    all_data = [
        (
            make_page_feats(feat2id, page),
            make_page_labels(label2id, labels),
        )
        for page, labels in pages
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
            stack = [
                self.embedding_layers[name](features.data[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors.data)
            inputs = torch.nn.utils.rnn.PackedSequence(
                torch.hstack(stack), features.batch_sizes
            )
        else:
            stack = [
                self.embedding_layers[name](features[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors)
            inputs = torch.hstack(stack)
        lstm_out, self.hidden_state = self.lstm_layer(inputs)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.output_layer(lstm_out)
        tag_space = tag_space.transpose(
            -1, 1
        )  # We need to transpose since it's a sequence (but why?!)
        return tag_space


class RNNCRF(nn.Module):
    def __init__(
        self,
        featdims,
        feat2id,
        veclen,
        id2label,
        label_weights,
        hidden_size=64,
        num_layer=1,
        bidirectional=True,
        dropout=0,
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
            dropout=dropout,
        )
        self.linear_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), len(id2label)
        )
        self.crf_layer = ConditionalRandomFieldWeightTrans(
            num_tags=len(id2label),
            label_weights=label_weights,
            constraints=allowed_transitions("BIO", dict(enumerate(id2label))),
        )

    def forward(
        self,
        features: PackedSequence | torch.Tensor,
        vectors: PackedSequence | torch.Tensor,
        mask: torch.Tensor,
    ):
        # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184
        if isinstance(features, PackedSequence):
            stack = [
                self.embedding_layers[name](features.data[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors.data)
            inputs = torch.nn.utils.rnn.PackedSequence(
                torch.hstack(stack), features.batch_sizes
            )
        else:
            stack = [
                self.embedding_layers[name](inputs[:, idx])
                for idx, name in enumerate(self.featdims)
            ]
            stack.append(vectors)
            inputs = torch.hstack(stack)
        lstm_out, self.hidden_state = self.lstm_layer(inputs)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.linear_layer(lstm_out)
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


class RNNSegmenteur:
    def __init__(self, model: PathLike = DEFAULT_RNN_MODEL, device="cpu"):
        model = Path(model)
        self.device = torch.device(device)
        with open(model.with_suffix(".json"), "rt") as infh:
            self.config = json.load(infh)
        self.model = RNN(**self.config)
        self.model.load_state_dict(torch.load(model))
        self.model.eval()
        self.model.to(device)

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for p in split_pages(words):
            page, _labels = make_rnn_features(p)
            features = make_page_feats(self.config["feat2id"], page)
            feats, vector = zip(*features)
            out = self.model(
                torch.LongTensor(feats, device=self.device),
                torch.FloatTensor(vector, device=self.device),
                torch.ones(len(feats), device=self.device),
            )
            for label_id, word in zip(out.argmax(-1).cpu(), p):
                word["segment"] = self.config["id2label"][label_id.item()]
                yield word
