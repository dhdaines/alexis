"""Entrainer un LSTM pour segmentation/identification"""

import argparse
import csv
import itertools
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from poutyne import EarlyStopping, ExponentialLR, Model, ModelCheckpoint, set_seeds
from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset

from scripts.train_rnn_crf import (
    make_all_data,
    make_dataset,
    pad_collate_fn,
    pad_collate_fn_predict,
)


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs", nargs="+", help="Fichiers CSV d'entrainement", type=Path
    )
    parser.add_argument(
        "--nepoch", default=100, type=int, help="Nombre maximal d'epochs d'entrainement"
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Taille du batch")
    parser.add_argument(
        "--word-dim", default=32, type=int, help="Dimension des embeddings des mots"
    )
    parser.add_argument(
        "--feat-dim", default=8, type=int, help="Dimension des embeddings des traits"
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="Facteur d'apprentissage"
    )
    parser.add_argument(
        "--gamma", default=0.99, type=float, help="Coefficient de reduction du LR"
    )
    parser.add_argument(
        "--bscale", default=1.0, type=float, help="Facteur applique aux debuts de bloc"
    )
    parser.add_argument(
        "--hidden-size", default=64, type=int, help="Largeur de la couche cachee"
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="Patience pour arret anticipe"
    )
    parser.add_argument("--seed", default=1381, type=int, help="Graine aléatoire")
    parser.add_argument(
        "--min-count",
        default=10,
        type=int,
        help="Seuil d'évaluation pour chaque classification",
    )
    parser.add_argument(
        "-x",
        "--cross-validation-folds",
        default=4,
        type=int,
        help="Faire la validation croisée pour évaluer le modèle.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Fichier destination pour modele",
        default="rnnmodel.pkl",
    )
    parser.add_argument(
        "-s",
        "--scores",
        help="Fichier destination pour évaluations",
        default="rnnscores.csv",
    )
    return parser


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
        self.output_layer = nn.Linear(
            hidden_size * (2 if bidirectional else 1), n_labels
        )

    def forward(
        self,
        features: PackedSequence | torch.Tensor,
        vectors: PackedSequence | torch.Tensor,
        _mask: torch.Tensor,
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
        tag_space = self.output_layer(lstm_out)
        tag_space = tag_space.transpose(
            -1, 1
        )  # We need to transpose since it's a sequence (but why?!)
        return tag_space


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cuda_device = 0
    device = torch.device(
        "cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu"
    )
    set_seeds(args.seed)

    X, y = zip(*make_dataset(args.csvs))
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
        "lower": args.word_dim,
        "rgb": args.feat_dim,
        "mctag": args.feat_dim,
        "uppercase": args.feat_dim,
        "title": args.feat_dim,
        "punc": args.feat_dim,
        "endpunc": args.feat_dim,
        "numeric": args.feat_dim,
        "bold": args.feat_dim,
        "italic": args.feat_dim,
        "toc": args.feat_dim,
        "header": args.feat_dim,
        "head:table": args.feat_dim,
        "head:chapitre": args.feat_dim,
        "head:annexe": args.feat_dim,
        "line:height": args.feat_dim,
        "line:indent": args.feat_dim,
        "line:gap": args.feat_dim,
        "first": args.feat_dim,
        "last": args.feat_dim,
    }
    all_data, feat2id, id2label = make_all_data(X, y, featdims, vecnames)
    kf = KFold(
        n_splits=args.cross_validation_folds, shuffle=True, random_state=args.seed
    )
    scores = {"test_macro_f1": []}
    label_counts = Counter(itertools.chain.from_iterable(y))
    labels = sorted(
        x for x in label_counts if x[0] == "B" and label_counts[x] >= args.min_count
    )
    veclen = len(all_data[0][0][0][1])
    for fold, (train_idx, dev_idx) in enumerate(kf.split(all_data)):
        train_data = Subset(all_data, train_idx)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=pad_collate_fn,
        )
        dev_data = Subset(all_data, dev_idx)
        dev_loader = DataLoader(
            dev_data, batch_size=args.batch_size, collate_fn=pad_collate_fn
        )

        my_network = MyNetwork(
            featdims, feat2id, veclen, len(id2label), hidden_size=args.hidden_size
        )
        optimizer = optim.Adam(my_network.parameters(), lr=args.lr)
        loss_function = nn.CrossEntropyLoss(
            # weight=torch.FloatTensor([10 if x[0] == "B" else 1 for x in id2label])
            # weight=torch.FloatTensor([label_weights.get(x, 1.0) for x in id2label])
            weight=torch.FloatTensor(
                [
                    (args.bscale if x[0] == "B" else 1.0) / label_counts[x]
                    for x in id2label
                ]
            )
        )
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
            epochs=args.nepoch,
            callbacks=[
                ExponentialLR(gamma=args.gamma),
                ModelCheckpoint(
                    monitor="val_fscore_macro",
                    filename=args.outfile,
                    mode="max",
                    save_best_only=True,
                    restore_best=True,
                    keep_only_last_best=True,
                    verbose=True,
                ),
                EarlyStopping(
                    monitor="val_fscore_macro",
                    mode="max",
                    patience=args.patience,
                    verbose=True,
                ),
            ],
        )
        ordering, sorted_test_data = zip(
            *sorted(enumerate(dev_data), reverse=True, key=lambda x: len(x[1][0]))
        )
        test_loader = DataLoader(
            sorted_test_data,
            batch_size=args.batch_size,  # FIXME: Actually not relevant here
            collate_fn=pad_collate_fn_predict,
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

    with open(args.scores, "wt") as outfh:
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
