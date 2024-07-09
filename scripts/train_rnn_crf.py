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
from bi_lstm_crf import CRF

from alexi import segment

DATA = list((Path(__file__).parent.parent / "data").glob("*.csv"))


class MyNetwork(nn.Module):
    def __init__(
        self,
        featdims,
        feat2id,
        veclen,
        n_labels,
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
        self.crf_layer = CRF(hidden_size * (2 if bidirectional else 1), n_labels)

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
        _scores, labels = self.crf_layer(lstm_out, mask)
        return lstm_out, labels, mask


class MyCRFLoss:
    def __init__(self, crf):
        self.crf = crf

    def __call__(self, returns, y_true):
        logits, _labels, mask = returns
        # Annoyingly, CRF requires masked labels to be valid indices
        y_true[~mask] = 0
        return self.crf.loss(logits, y_true, mask)


def my_accuracy(y_pred, y_true):
    """poutyne's "accuracy" is totally useless here"""
    _score, y_pred, y_mask = y_pred
    n_labels = 0
    n_true = 0
    for pred, true, _mask in zip(y_pred, y_true, y_mask):
        n_labels += len(pred)  # it is not padded
        n_true += torch.eq(torch.Tensor(pred), true[: len(pred)].cpu()).sum()
    return n_true / n_labels * 100


def main():
    cuda_device = 0
    device = torch.device(
        "cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu"
    )
    batch_size = 32
    seed = 1381
    set_seeds(seed)

    X, y = zip(*make_dataset(DATA))
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
        "rgb": 8,
        "mctag": 8,
        "uppercase": 8,
        "title": 8,
        "punc": 8,
        "endpunc": 8,
        "numeric": 8,
        "bold": 8,
        "italic": 8,
        "toc": 8,
        "header": 8,
        "head:table": 8,
        "head:chapitre": 8,
        "head:annexe": 8,
        "line:height": 8,
        "line:indent": 8,
        "line:gap": 8,
        "first": 8,
        "last": 8,
    }
    all_data, feat2id, id2label = make_all_data(X, y, featdims, vecnames)
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    scores = {"test_macro_f1": []}
    label_counts = Counter(itertools.chain.from_iterable(y))
    labels = sorted(x for x in label_counts if x[0] == "B" and label_counts[x] >= 10)
    veclen = len(all_data[0][0][0][1])
    for fold, (train_idx, dev_idx) in enumerate(kf.split(all_data)):
        train_data = Subset(all_data, train_idx)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
        )
        dev_data = Subset(all_data, dev_idx)
        dev_loader = DataLoader(
            dev_data, batch_size=batch_size, collate_fn=pad_collate_fn
        )

        my_network = MyNetwork(featdims, feat2id, veclen, len(id2label), hidden_size=80)
        optimizer = optim.Adam(my_network.parameters(), lr=0.1)
        loss_function = MyCRFLoss(my_network.crf_layer)
        model = Model(
            my_network,
            optimizer,
            loss_function,
            batch_metrics=[my_accuracy],
            device=device,
        )
        model.fit_generator(
            train_loader,
            dev_loader,
            epochs=100,
            callbacks=[
                ExponentialLR(gamma=0.9),
                ModelCheckpoint(
                    monitor="val_my_accuracy",
                    filename="rnnmodel.pkl",
                    mode="max",
                    save_best_only=True,
                    restore_best=True,
                    keep_only_last_best=True,
                    verbose=True,
                ),
                EarlyStopping(
                    monitor="val_my_accuracy", mode="max", patience=10, verbose=True
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
            _scores, tags, _mask = batch
            for length, row in zip(lengths, tags):
                predictions.append(np.array(row[:length]))
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


if __name__ == "__main__":
    main()
