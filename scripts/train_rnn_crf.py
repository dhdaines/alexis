"""Entrainer un LSTM pour segmentation/identification"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from poutyne import (
    EarlyStopping,
    ExponentialLR,
    Model,
    ModelCheckpoint,
    set_seeds,
    Accuracy,
    F1,
)
from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics
from torch.utils.data import DataLoader, Subset

from alexi.segment import make_rnn_data, pad_collate_fn, pad_collate_fn_predict, RNNCRF
from tokenizers import Tokenizer


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs", nargs="+", help="Fichiers CSV d'entrainement", type=Path
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device pour rouler l'entrainement"
    )
    parser.add_argument(
        "--nepoch", default=45, type=int, help="Nombre maximal d'epochs d'entrainement"
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
        "--hidden-size", default=80, type=int, help="Largeur de la couche cachee"
    )
    parser.add_argument("--early-stopping", action="store_true", help="Arret anticipe")
    parser.add_argument(
        "--patience", default=10, type=int, help="Patience pour arret anticipe"
    )
    parser.add_argument("--seed", default=1381, type=int, help="Graine aléatoire")
    parser.add_argument(
        "--features", default="text+layout+structure", help="Extracteur de traits"
    )
    parser.add_argument("--labels", default="literal", help="Transformateur de classes")
    parser.add_argument(
        "--min-count",
        default=10,
        type=int,
        help="Seuil d'évaluation pour chaque classification",
    )
    parser.add_argument(
        "-x",
        "--cross-validation-folds",
        default=1,
        type=int,
        help="Faire la validation croisée pour évaluer le modèle si plus que 1.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Fichier destination pour modele",
        type=Path,
        default="rnn_crf.pt",
    )
    parser.add_argument(
        "-s",
        "--scores",
        help="Fichier destination pour évaluations",
    )
    parser.add_argument(
        "-t", "--tokenize", action="store_true", help="Tokeniser les mots"
    )
    parser.add_argument(
        "-i",
        "--init-model",
        help="Fichier modele pour initialisation",
        default="rnn.pt",
        type=Path,
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Geler les parametres initialisees"
    )
    parser.add_argument(
        "--constrain", action="store_true", help="Interdire les transitions impossibles"
    )
    return parser


class MyCRFLoss:
    def __init__(self, crf):
        self.crf = crf

    def __call__(self, returns, y_true):
        logits, _labels, mask = returns
        # CRF requires masked labels to be valid indices
        y_true[~mask] = 0
        return -self.crf(logits, y_true, mask)


def CRFMetric(cls, *args, **kwargs):
    metric = cls(*args, **kwargs)

    def crf_forward(y_pred, y_true):
        _score, preds, mask = y_pred
        # For some unknown reason poutyne expects logits/probabilities and not labels here
        n_labels = y_true.max() + 1
        fake_logits = torch.zeros(y_true.shape + (n_labels,), device=y_true.device)
        for idx, labels in enumerate(preds):
            fake_logits[idx, range(len(labels)), labels] = 1.0
        # Have to set masked labels back to the ignore index
        y_true[~mask] = metric.ignore_index
        # Also the shape is weird because pytorch is weird
        return cls.forward(metric, fake_logits.transpose(1, -1), y_true)

    metric.forward = crf_forward
    return metric


def freeze_rnn(my_network):
    for layer in my_network.embedding_layers.values():
        layer.requires_grad_(False)
    my_network.lstm_layer.requires_grad_(False)
    my_network.output_layer.requires_grad_(False)


def run_cv(args, all_data, featdims, feat2id, label_counts, label_weights, id2label):
    kf = KFold(
        n_splits=args.cross_validation_folds, shuffle=True, random_state=args.seed
    )
    scores = {"test_macro_f1": []}
    if args.labels == "iobonly":
        eval_labels = sorted(label_counts.keys())
    else:
        eval_labels = sorted(
            x for x in label_counts if x[0] == "B" and label_counts[x] >= args.min_count
        )
    veclen = len(all_data[0][0][0][1])
    device = torch.device(args.device)

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
        config = {
            "feat2id": feat2id,
            "id2label": id2label,
            "featdims": featdims,
            "veclen": veclen,
            "label_weights": label_weights,
            "hidden_size": args.hidden_size,
            "features": args.features,
            "labels": args.labels,
            "constrain": args.constrain,
        }
        foldfile = args.outfile.with_stem(args.outfile.stem + f"_fold{fold+1}")
        with open(foldfile.with_suffix(".json"), "wt", encoding="utf-8") as outfh:
            json.dump(config, outfh, ensure_ascii=False, indent=2)
        my_network = RNNCRF(**config)
        if args.init_model:
            fold_init = args.init_model.with_stem(
                args.init_model.stem + f"_fold{fold+1}"
            )
            state_dict = torch.load(fold_init)
            my_network.load_state_dict(state_dict, strict=False)
            if args.freeze:
                freeze_rnn(my_network)
        optimizer = optim.Adam(my_network.parameters(), lr=args.lr)
        loss_function = MyCRFLoss(my_network.crf_layer)
        model = Model(
            my_network,
            optimizer,
            loss_function,
            batch_metrics=[CRFMetric(Accuracy), CRFMetric(F1)],
            device=device,
        )
        callbacks = [ExponentialLR(gamma=args.gamma)]
        if args.early_stopping:
            callbacks.append(
                ModelCheckpoint(
                    monitor="val_fscore_macro",
                    filename=str(foldfile),
                    mode="max",
                    save_best_only=True,
                    restore_best=True,
                    keep_only_last_best=True,
                    verbose=True,
                )
            )
            callbacks.append(
                EarlyStopping(
                    monitor="val_fscore_macro",
                    mode="max",
                    patience=args.patience,
                    verbose=True,
                )
            )
        model.fit_generator(
            train_loader,
            dev_loader,
            epochs=args.nepoch,
            callbacks=callbacks,
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
        for batch in out:
            _logits, tags, _mask = batch
            predictions.extend(tags)
        y_pred = [[id2label[x] for x in page] for page in predictions]
        y_true = [[id2label[x] for x in page] for _, page in sorted_test_data]
        macro_f1 = metrics.flat_f1_score(
            y_true, y_pred, labels=eval_labels, average="macro", zero_division=0.0
        )
        scores["test_macro_f1"].append(macro_f1)
        print("fold", fold + 1, "ALL", macro_f1)
        for name in eval_labels:
            label_f1 = metrics.flat_f1_score(
                y_true, y_pred, labels=[name], average="micro", zero_division=0.0
            )
            scores.setdefault(name, []).append(label_f1)
            print("fold", fold + 1, name, label_f1)
        if not args.early_stopping:
            torch.save(my_network.state_dict(), foldfile)

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
        for name in eval_labels:
            row = makerow(name, scores[name])
            writer.writerow(row)
            print("average", row["Label"], row["Average"])


def run_training(args, train_data, featdims, feat2id, label_weights, id2label):
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    veclen = len(train_data[0][0][0][1])
    device = torch.device(args.device)
    config = {
        "feat2id": feat2id,
        "id2label": id2label,
        "featdims": featdims,
        "veclen": veclen,
        "label_weights": label_weights,
        "hidden_size": args.hidden_size,
        "features": args.features,
        "labels": args.labels,
        "constrain": args.constrain,
    }
    with open(args.outfile.with_suffix(".json"), "wt", encoding="utf-8") as outfh:
        json.dump(config, outfh, ensure_ascii=False, indent=2)
    my_network = RNNCRF(**config)
    if args.init_model:
        state_dict = torch.load(args.init_model)
        my_network.load_state_dict(state_dict, strict=False)
        if args.freeze:
            freeze_rnn(my_network)
    optimizer = optim.Adam(my_network.parameters(), lr=args.lr)
    loss_function = MyCRFLoss(my_network.crf_layer)
    model = Model(
        my_network,
        optimizer,
        loss_function,
        device=device,
    )
    model.fit_generator(
        train_loader,
        epochs=args.nepoch,
        callbacks=[
            ExponentialLR(gamma=args.gamma),
        ],
    )
    torch.save(my_network.state_dict(), args.outfile)


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seeds(args.seed)
    if args.scores is None:
        args.scores = args.outfile.with_suffix(".csv")
    tokenizer = None
    if args.tokenize:
        tokenizer = Tokenizer.from_pretrained("camembert-base")

    all_data, featdims, feat2id, label_counts, id2label = make_rnn_data(
        args.csvs, features=args.features, labels=args.labels, tokenizer=tokenizer
    )
    # Note that weights must be greater than 1.0 for training to work
    label_weights = [
        (args.bscale if x[0] == "B" else 1.0) * math.exp(1.0 / label_counts[x])
        for x in id2label
    ]

    print("Vocabulary size:")
    for feat, vals in feat2id.items():
        print(f"\t{feat}: {len(vals)}")

    if args.cross_validation_folds == 1:
        run_training(args, all_data, featdims, feat2id, label_weights, id2label)
    else:
        run_cv(args, all_data, featdims, feat2id, label_counts, label_weights, id2label)


if __name__ == "__main__":
    main()
