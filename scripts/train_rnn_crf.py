import csv
import itertools
from collections import Counter
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

from alexi import segment

DATA = list((Path(__file__).parent.parent / "data").glob("*.csv"))


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


def make_page_feats(feat2id, page, featdims, vecnames):
    return [
        (
            [feat2id[name][feats[name]] for name in featdims],
            [float(feats[name]) for name in vecnames],
        )
        for feats in page
    ]


def make_page_labels(label2id, page):
    return [label2id[tag] for tag in page]


def make_all_data(X, y, featdims, vecnames):
    labelset = set(itertools.chain.from_iterable(y))
    id2label = sorted(labelset, reverse=True)
    label2id = dict((label, idx) for (idx, label) in enumerate(id2label))
    feat2id = {name: {"": 0} for name in featdims}
    for feats in itertools.chain.from_iterable(X):
        for name, ids in feat2id.items():
            if feats[name] not in ids:
                ids[feats[name]] = len(ids)
    print("Vocabulary size:")
    for feat, vals in feat2id.items():
        print(f"\t{feat}: {len(vals)}")

    all_data = [
        (
            make_page_feats(feat2id, page, featdims, vecnames),
            make_page_labels(label2id, labels),
        )
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
    return all_data, feat2id, id2label


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
    label_norm = min(label_counts.values())
    # Label weights must be at least 1.0 for learning to work (for whatever reason)
    label_weights = [1.0 + label_norm / label_counts[x] for x in id2label]
    # label_weights = [1.0 for x in id2label]
    labels = sorted(x for x in label_counts if x[0] == "B" and label_counts[x] >= 10)
    veclen = len(all_data[0][0][0][1])
    for fold, (train_idx, dev_idx) in enumerate(kf.split(all_data)):
        train_data = Subset(all_data, train_idx)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=segment.pad_collate_fn,
        )
        dev_data = Subset(all_data, dev_idx)
        dev_loader = DataLoader(
            dev_data, batch_size=batch_size, collate_fn=segment.pad_collate_fn
        )

        my_network = segment.RNNCRF(
            featdims,
            feat2id,
            veclen,
            id2label,
            label_weights=label_weights,
            hidden_size=80,
        )
        optimizer = optim.Adam(my_network.parameters(), lr=0.01)
        loss_function = MyCRFLoss(my_network.crf_layer)
        model = Model(
            my_network,
            optimizer,
            loss_function,
            batch_metrics=[CRFMetric(Accuracy), CRFMetric(F1)],
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
                    filename="rnncrf.pt",
                    mode="max",
                    save_best_only=True,
                    restore_best=True,
                    keep_only_last_best=True,
                    verbose=True,
                ),
                EarlyStopping(
                    monitor="val_fscore_macro",
                    mode="max",
                    patience=10,
                    verbose=True,
                ),
            ],
        )
        ordering, sorted_test_data = zip(
            *sorted(enumerate(dev_data), reverse=True, key=lambda x: len(x[1][0]))
        )
        test_loader = DataLoader(
            sorted_test_data,
            batch_size=batch_size,
            collate_fn=segment.pad_collate_fn_predict,
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
