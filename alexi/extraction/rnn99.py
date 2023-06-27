import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

MODELPATH = Path(__file__).with_suffix("")
INDEXLIKE = re.compile(r"^(\d+|[a-z])[\)\.]|-$")


def load_model():
    model = keras.models.load_model(MODELPATH)
    with open(MODELPATH.with_suffix(".scaler.pkl"), "rb") as infh:
        scaler = pickle.load(infh)
    with open(MODELPATH.with_suffix(".vocab.json"), "rt") as infh:
        vocab = json.load(infh)
    return model, scaler, vocab


def load_csv(path):
    df = pd.read_csv(path)
    df = df.assign(
        xdelta=df.x0.diff(),
        ydelta=df.doctop.diff(),
        height=df.bottom - df.top,
        width=df.x1 - df.x0,
    )
    df = df.assign(xdd=df.xdelta.diff(), ydd=df.ydelta.diff())
    df = df.fillna(0)
    return df


def make_scaler(df):
    scaler = StandardScaler()
    scaler.fit(df.loc[:, ["x0", "top", "width", "height"]])
    return scaler


def make_features(df, scaler):
    feats = df.loc[
        :, ["xdelta", "ydelta", "x0", "top", "xdd", "ydd", "width", "height"]
    ]
    index = []
    idelta = []
    isnum = []
    iseq = []
    chapitre = []
    section = []
    prevnum = 0
    for t in df.loc[:, "text"]:
        t = str(t)  # fuck you, pandas
        isnum.append(t.isnumeric())
        chapitre.append("chap" in t.lower())
        section.append("sect" in t.lower())
        if m := INDEXLIKE.match(t):
            index.append(True)
            if m.group(1) and m.group(1).isnumeric():
                idx = int(m.group(1))
                idelta.append(idx - prevnum)
                iseq.append(idx == prevnum + 1)
                prevnum = idx
            else:
                idelta.append(0)
                iseq.append(False)
        else:
            index.append(False)
            idelta.append(0)
            iseq.append(False)
    feats = feats.assign(xdd=np.sign(df.loc[:, "xdd"]), ydd=np.sign(df.loc[:, "ydd"]))
    scaled = scaler.transform(df.loc[:, ["x0", "top", "width", "height"]])
    feats = feats.assign(
        x0=scaled[:, 0], top=scaled[:, 1], width=scaled[:, 2], height=scaled[:, 3]
    )
    feats = feats.assign(
        index=index,
        idelta=idelta,
        isnum=isnum,
        iseq=iseq,
        chapitre=chapitre,
        section=section,
    )
    return feats


def make_targets(df):
    return df["tag"]


def make_blocks(seq, dtype="int32", blocksize=128):
    blocks = [
        seq[start : start + blocksize] for start in range(0, seq.shape[0], blocksize)
    ]
    return keras.utils.pad_sequences(blocks, padding="post", dtype=dtype)


def make_model(nfeats, vocab, blocksize=128):
    input_features = layers.Input(name="features", shape=(blocksize, nfeats))
    lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_features)
    dense = layers.Dense(32, activation="relu")(lstm)
    outputs = layers.Dense(
        len(vocab), activation="softmax", name="predictions"
    )(dense)
    model = keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )


def train_model(model, train_df, devel_df, blocksize=128):
    scaler = make_scaler(train_df)
    features = make_features(train_df, scaler)
    targets = make_targets(train_df)
    lookup = tf.keras.layers.StringLookup()
    lookup.adapt(targets)

    xf_train_blocks = make_blocks(features, "float32", blocksize)
    y_train_blocks = make_blocks(lookup(targets), blocksize=blocksize)

    xf_devel_blocks = make_blocks(make_features(devel_df), "float32", blocksize)
    y_devel_blocks = make_blocks(lookup(make_targets(devel_df)), blocksize=blocksize)
    callback = tf.keras.callbacks.EarlyStopping(
        start_from_epoch=50,
        monitor="val_sparse_categorical_accuracy",
        patience=25,
        restore_best_weights=True,
    )
    model = make_model(features.shape[1], lookup.vocabulary_size)
    model.fit(
        xf_train_blocks,
        y_train_blocks,
        batch_size=16,
        epochs=1000,
        validation_data=(xf_devel_blocks, y_devel_blocks),
        callbacks=(callback,),
    )
    return model, scaler, lookup.get_vocabulary()


def chunk(df, model, scaler, vocab):
    xf_blocks = make_blocks(make_features(df, scaler), "float32")
    predictions = [
        vocab[i] for i in np.concatenate(model.predict(xf_blocks, verbose=0).argmax(axis=2))
    ]
    words = df.iloc[:, 1]
    sentences = []
    chunk = []
    tag = None
    for label, word in zip(predictions, words):
        if label[0] == "B":
            if chunk:
                sentences.append((tag, " ".join(str(x) for x in chunk)))
                chunk = []
            tag = label.partition("-")[2]
        elif len(chunk) == 0:
            tag = label.partition("-")[2]
        chunk.append(word)
    sentences.append((tag, " ".join(chunk)))
    return sentences


def tag(df, model, scaler, vocab):
    xf_blocks = make_blocks(make_features(df, scaler), "float32")
    predictions = [
        vocab[i] for i in np.concatenate(model.predict(xf_blocks, verbose=0).argmax(axis=2))
    ]
    return predictions[: df.shape[0]]


def main(args):
    model, scaler, vocab = load_model()
    df = load_csv(args.csv)
    df = df.assign(tag=tag(df, model, scaler, vocab))
    df.to_csv(sys.stdout, index=False)


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Fichier CSV d'entree", type=Path)
    return parser


if __name__ == "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
