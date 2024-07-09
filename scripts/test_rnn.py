"""Tester un RNN"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from poutyne import Model
from sklearn_crfsuite import metrics
from torch.utils.data import DataLoader

from alexi.segment import load_rnn_data, pad_collate_fn_predict, RNN


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", help="Fichier modele", default="rnn.model", type=Path
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device pour rouler la prediction"
    )
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV de test", type=Path)
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.model.with_suffix(".json"), "rt") as infh:
        config = json.load(infh)
        id2label = config["id2label"]
        feat2id = config["feat2id"]
    all_data = load_rnn_data(args.csvs, feat2id, id2label)
    ordering, sorted_test_data = zip(
        *sorted(enumerate(all_data), reverse=True, key=lambda x: len(x[1][0]))
    )
    test_loader = DataLoader(
        sorted_test_data,
        batch_size=32,
        collate_fn=pad_collate_fn_predict,
    )
    device = torch.device(args.device)
    my_network = torch.load(args.model)
    # FIXME: Poutyne considered not useful for prediction
    model = Model(
        my_network,
        optim.Adam(my_network.parameters()),  # FIXME: Obviously not used
        nn.CrossEntropyLoss(),
        device=device,
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
    eval_labels = sorted(
        ["O", *(c for c in id2label if c.startswith("B-"))],
        key=lambda name: (name[1:], name[0]),
    )
    report = metrics.flat_classification_report(y_true, y_pred, labels=eval_labels)
    print(report)


if __name__ == "__main__":
    main()
