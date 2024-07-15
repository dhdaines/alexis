"""Tester un RNN"""

import argparse
import json
import logging
from pathlib import Path

import torch
from sklearn_crfsuite import metrics
from torch.utils.data import DataLoader

from alexi.segment import load, load_rnn_data, pad_collate_fn_predict, RNNCRF


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", help="Fichier modele", default="rnn.pt", type=Path
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
        id2label = config["labels"]
        feat2id = config["feat2id"]
    all_data = load_rnn_data(load(args.csvs), feat2id, id2label)
    ordering, sorted_test_data = zip(
        *sorted(enumerate(all_data), reverse=True, key=lambda x: len(x[1][0]))
    )
    test_loader = DataLoader(
        sorted_test_data,
        batch_size=32,
        collate_fn=pad_collate_fn_predict,
    )
    device = torch.device(args.device)
    model = RNNCRF(**config)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.to(device)
    predictions = []
    for batch in test_loader:
        _logits, labels, _mask = model(*(t.to(device) for t in batch))
        predictions.extend(labels)
    y_pred = [[id2label[x] for x in page] for page in predictions]
    y_true = [[id2label[x] for x in page] for _, page in sorted_test_data]
    eval_labels = sorted(
        ["O", *(c for c in id2label if c.startswith("B-"))],
        key=lambda name: (name[1:], name[0]),
    )
    report = metrics.flat_classification_report(
        y_true, y_pred, labels=eval_labels, zero_division=0
    )
    print(report)


if __name__ == "__main__":
    main()
