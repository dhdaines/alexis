import csv
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from poutyne import EarlyStopping, ExponentialLR, Model, ModelCheckpoint, set_seeds
from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from torch.utils.data import DataLoader, Subset

from alexi import segment

DATA = list(Path("data").glob("*.csv"))


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


def make_dataset(csvs):
    iobs = segment.load(csvs)
    for p in segment.split_pages(segment.filter_tab(iobs)):
        features = list(
            dict(w.split("=", maxsplit=2) for w in feats)
            for feats in segment.textpluslayoutplusstructure_features(p)
        )
        for f, w in zip(features, p):
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
        labels = list(segment.page2labels(p))
        yield features, labels


X, y = zip(*make_dataset(DATA))

labelset = set(itertools.chain.from_iterable(y))
id2label = sorted(labelset, reverse=True)
label2id = dict((label, idx) for (idx, label) in enumerate(id2label))

vecnames = [
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
featdims = {
    "lower": 32,
    "rgb": 4,
    "mctag": 4,
    "uppercase": 4,
    "title": 4,
    "punc": 4,
    "endpunc": 4,
    "numeric": 4,
    "bold": 4,
    "italic": 4,
    "toc": 4,
    "header": 4,
    "head:table": 4,
    "head:chapitre": 4,
    "head:annexe": 4,
    "line:height": 4,
    "line:indent": 4,
    "line:gap": 4,
    "first": 4,
    "last": 4,
}

feat2id = {name: {"": 0} for name in featdims}
for feats in itertools.chain.from_iterable(X):
    for name, ids in feat2id.items():
        if feats[name] not in ids:
            ids[feats[name]] = len(ids)
print("Vocabulary size:")
for feat, vals in feat2id.items():
    print(f"\t{feat}: {len(vals)}")


def make_page_feats(feat2id, page):
    return [
        (
            [feat2id[name][feats[name]] for name in featdims],
            [float(feats[name]) for name in vecnames],
        )
        for feats in page
    ]


def make_page_labels(label2id, page):
    return [label2id[tag] for tag in page]


all_data = [
    (make_page_feats(feat2id, page), make_page_labels(label2id, labels))
    for page, labels in zip(X, y)
]
veclen = len(all_data[0][0][0][1])
vecmax = np.zeros(veclen)
for page, _ in all_data:
    for _, vector in page:
        vecmax = np.maximum(vecmax, np.abs(vector))
# print("Scaling:")
# for feat, val in zip(vecnames, vecmax):
#     print(f"\t{feat}: {val}")


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
    return (
        (pack_padded_sequences_features, pack_padded_sequences_vectors),
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
    return (pack_padded_sequences_features, pack_padded_sequences_vectors)


class MyNetwork(nn.Module):
    def __init__(
        self,
        featdims,
        feat2id,
        n_labels,
        hidden_size=64,
        num_layer=1,
        bidirectional=True,
        dropout=0,
    ):
        super().__init__()
        self.hidden_state = None
        self.embedding_layers = {}
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
        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), n_labels
        )

    def forward(
        self,
        features: PackedSequence | torch.Tensor,
        vectors: PackedSequence | torch.Tensor,
    ):
        # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184
        if isinstance(features, PackedSequence):
            stack = [
                self.embedding_layers[name](features.data[:, idx])
                for idx, name in enumerate(featdims)
            ]
            stack.append(vectors.data)
            inputs = torch.nn.utils.rnn.PackedSequence(
                torch.hstack(stack), features.batch_sizes
            )
        else:
            stack = [
                self.embedding_layers[name](inputs[:, idx])
                for idx, name in enumerate(featdims)
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


cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
batch_size = 32
seed = 1381
set_seeds(seed)

kf = KFold(n_splits=4, shuffle=True, random_state=seed)
scores = {"test_macro_f1": []}
label_counts = Counter(itertools.chain.from_iterable(y))
labels = sorted(x for x in label_counts if x[0] == "B" and label_counts[x] >= 10)
for fold, (train_idx, dev_idx) in enumerate(kf.split(all_data)):
    train_data = Subset(all_data, train_idx)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    dev_data = Subset(all_data, dev_idx)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=pad_collate_fn)

    my_network = MyNetwork(featdims, feat2id, len(id2label))
    optimizer = optim.Adam(my_network.parameters(), lr=0.1)
    loss_function = nn.CrossEntropyLoss()
    model = Model(
        my_network,
        optimizer,
        loss_function,
        batch_metrics=["accuracy", "f1"],
        device=device,
    )
    model.fit_generator(
        train_loader,
        dev_loader,
        epochs=100,
        callbacks=[
            ExponentialLR(gamma=0.99),
            ModelCheckpoint(
                monitor="val_fscore_macro",
                filename="rnnmodel.pkl",
                mode="max",
                save_best_only=True,
                restore_best=True,
                keep_only_last_best=True,
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_fscore_macro", mode="max", patience=10, verbose=True
            ),
        ],
    )
    ordering, sorted_test_data = zip(
        *sorted(enumerate(dev_data), reverse=True, key=lambda x: len(x[1][0]))
    )
    test_loader = DataLoader(
        sorted_test_data, batch_size=batch_size, collate_fn=pad_collate_fn_predict
    )
    out = model.predict_generator(test_loader, concatenate_returns=False)
    predictions = []
    lengths = [len(tokens) for tokens, _ in sorted_test_data]
    for batch in out:
        # numpy.transpose != torch.transpose because Reasons
        batch = batch.transpose((0, 2, 1)).argmax(-1)
        for length, row in zip(lengths, batch):
            predictions.append(row[:length])
        del lengths[: len(batch)]
    y_pred = [[id2label[x] for x in page] for page in predictions]
    y_true = [[id2label[x] for x in page] for _, page in sorted_test_data]
    macro_f1 = metrics.flat_f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0.0
    )
    scores["test_macro_f1"].append(macro_f1)
    print("fold", fold + 1, "ALL", macro_f1)
    for name in labels:
        label_f1 = metrics.flat_f1_score(
            y_true, y_pred, labels=[name], average="micro", zero_division=0.0
        )
        scores.setdefault(name, []).append(label_f1)
        print("fold", fold + 1, name, label_f1)

with open("rnnscores.csv", "wt") as outfh:
    fieldnames = [
        "Label",
        "Average",
        *range(1, len(scores["test_macro_f1"]) + 1),
    ]
    writer = csv.DictWriter(outfh, fieldnames=fieldnames)
    writer.writeheader()

    def makerow(name, scores):
        row = {"Label": name, "Average": np.mean(scores)}
        for idx, score in enumerate(scores):
            row[idx + 1] = score
        return row

    row = makerow("ALL", scores["test_macro_f1"])
    writer.writerow(row)
    print("average", "ALL", row["Average"])
    for name in labels:
        row = makerow(name, scores[name])
        writer.writerow(row)
        print("average", row["Label"], row["Average"])
