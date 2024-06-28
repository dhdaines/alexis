import csv
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from poutyne import EarlyStopping, ExponentialLR, Model, set_seeds
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


def make_dataset(csvs):
    iobs = segment.load(csvs)
    for p in segment.split_pages(segment.filter_tab(iobs)):
        features = list(
            dict(x.split("=", maxsplit=2) for x in feats)
            for feats in segment.textpluslayoutplusstructure_features(p)
        )
        labels = list(segment.page2labels(p))
        yield features, labels


def pad_collate_fn(batch):
    sequences_tokens, sequences_labels, lengths = zip(
        *[
            (torch.LongTensor(tokens), torch.LongTensor(labels), len(tokens))
            for (tokens, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
        ]
    )
    lengths = torch.LongTensor(lengths)
    padded_sequences_tokens = pad_sequence(
        sequences_tokens, batch_first=True, padding_value=0
    )
    pack_padded_sequences_tokens = pack_padded_sequence(
        padded_sequences_tokens, lengths.cpu(), batch_first=True
    )
    padded_sequences_labels = pad_sequence(
        sequences_labels, batch_first=True, padding_value=-100
    )
    return pack_padded_sequences_tokens, padded_sequences_labels


def pad_collate_fn_predict(batch):
    # Require data to be externally sorted by length for prediction
    # (otherwise we have no idea which output corresponds to which input! WTF Poutyne!)
    sequences_tokens, lengths = zip(
        *((torch.LongTensor(tokens), len(tokens)) for (tokens, labels) in batch)
    )
    lengths = torch.LongTensor(lengths)
    padded_sequences_tokens = pad_sequence(
        sequences_tokens, batch_first=True, padding_value=0
    )
    pack_padded_sequences_tokens = pack_padded_sequence(
        padded_sequences_tokens, lengths.cpu(), batch_first=True
    )
    return pack_padded_sequences_tokens


class MyNetwork(nn.Module):
    def __init__(
        self,
        featnames,
        featdims,
        feat2id,
        n_labels,
        hidden_size=32,
        num_layer=1,
        bidirectional=True,
    ):
        super().__init__()
        self.hidden_state = None
        self.embedding_layers = {}
        for name in featnames:
            self.embedding_layers[name] = nn.Embedding(
                len(feat2id[name]),
                featdims[name],
                padding_idx=0,
            )
            self.add_module(f"embedding_{name}", self.embedding_layers[name])
        dimension = sum(featdims.values())
        self.lstm_layer = nn.LSTM(
            input_size=dimension,
            hidden_size=hidden_size,
            num_layers=num_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), n_labels
        )

    def forward(self, pack_padded_tokens_vectors: PackedSequence | torch.Tensor):
        # https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184
        if isinstance(pack_padded_tokens_vectors, PackedSequence):
            embeddings = torch.hstack(
                [
                    self.embedding_layers[name](pack_padded_tokens_vectors.data[:, idx])
                    for idx, name in enumerate(featnames)
                ]
            )
            embeddings = torch.nn.utils.rnn.PackedSequence(
                embeddings, pack_padded_tokens_vectors.batch_sizes
            )
        else:
            embeddings = torch.hstack(
                [
                    self.embedding_layers[name](pack_padded_tokens_vectors[:, idx])
                    for idx, name in enumerate(featnames)
                ]
            )
        lstm_out, self.hidden_state = self.lstm_layer(embeddings)
        if isinstance(lstm_out, PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.output_layer(lstm_out)
        tag_space = tag_space.transpose(
            -1, 1
        )  # We need to transpose since it's a sequence (but why?!)
        return tag_space


X, y = zip(*make_dataset(DATA))

labelset = set(itertools.chain.from_iterable(y))
id2label = sorted(labelset, reverse=True)
label2id = dict((label, idx) for (idx, label) in enumerate(id2label))

featnames = sorted(k for k in X[0][0] if not k.startswith("line:"))
featdims = {name: 32 if name in ("text", "lower") else 2 for name in featnames}
feat2id = {name: {"": 0} for name in featnames}
for feats in itertools.chain.from_iterable(X):
    for name, ids in feat2id.items():
        if feats[name] not in ids:
            ids[feats[name]] = len(ids)

all_data = [
    (
        [[feat2id[name][feats[name]] for name in featnames] for feats in page],
        [label2id[tag] for tag in labels],
    )
    for page, labels in zip(X, y)
]

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
batch_size = 32
seed = 1381
set_seeds(seed)

kf = KFold(n_splits=4, shuffle=True, random_state=seed)
scores = {"test_macro_f1": []}
label_counts = Counter(itertools.chain.from_iterable(y))
labels = [x for x in label_counts if x[0] == "B" and label_counts[x] >= 10]
for fold, (train_idx, dev_idx) in enumerate(kf.split(all_data)):
    train_data = Subset(all_data, train_idx)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    dev_data = Subset(all_data, dev_idx)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=pad_collate_fn)

    my_network = MyNetwork(featnames, featdims, feat2id, len(id2label))
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
            EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=True),
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

    writer.writerow(makerow("ALL", scores["test_macro_f1"]))
    for name in labels:
        writer.writerow(makerow(name, scores[name]))
