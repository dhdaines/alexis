import argparse
import fasttext
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore

MODELPATH = Path(__file__).with_suffix("")
INDEXLIKE = re.compile(r"^(\d+|[a-z])[\)\.]|[•-]$")


def load_model(path: Optional[Path] = None):
    if path is None:
        path = MODELPATH
    model = keras.models.load_model(path)
    with open(path.with_suffix(".scaler.pkl"), "rb") as infh:
        scaler = pickle.load(infh)
    with open(path.with_suffix(".vocab.json"), "rt") as infh:
        vocab = json.load(infh)
    wordvecs = fasttext.load_model(str(MODELPATH.with_suffix(".fasttext")))
    return model, scaler, vocab, wordvecs


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
    scaler.fit(df.loc[:, ["x0", "width", "height"]])
    return scaler


def make_features(df, scaler):
    # The only features we use unmodified!
    feats = df.loc[:, ["xdelta", "ydelta"]]
    index = []
    idelta = []
    isnum = []
    iseq = []
    prevnum = 0
    for t in df.loc[:, "text"]:
        t = str(t)
        isnum.append(t.isnumeric())
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
    scaled = scaler.transform(df.loc[:, ["x0", "width", "height"]])
    feats = feats.assign(
        x0=scaled[:, 0], width=scaled[:, 1], height=scaled[:, 2]
    )
    feats = feats.assign(
        index=index,
        idelta=idelta,
        isnum=isnum,
        iseq=iseq,
    )
    return feats


def simplify_targets(tag):
    """Enlever certains tags difficiles a predire"""
    if 'Tableau' in tag:
        return 'O'  # They just do not work, have to do them elsewhere
    if 'TOC' in tag:
        return 'O'  # Begone
    elif tag[0] == 'I':
        return tag.partition('-')[0]
    elif tag == 'B-Amendement':
        return 'B-Alinea'
    elif tag == 'B-SousSection':
        return'B-Section'
    elif tag == 'B-Annexe':
        return'B-Chapitre'
    return tag


def make_targets(df):
    return df["tag"].apply(simplify_targets)


def make_words(df, ft):
    return np.array([ft.get_word_vector(str(w)) for w in df["text"]])


def make_blocks(seq, dtype="int32", blocksize=128):
    blocks = [
        seq[start : start + blocksize] for start in range(0, seq.shape[0], blocksize)
    ]
    return keras.utils.pad_sequences(blocks, padding="post", dtype=dtype)


def make_model(nfeats: int, ntags: int, ndim: int, blocksize=128):
    input_features = layers.Input(name="features", shape=(blocksize, nfeats))
    input_words = layers.Input(name="words", shape=(blocksize, ndim))
    concat = layers.concatenate([input_features, input_words])
    lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(concat)
    outputs = layers.Dense(
        ntags, activation="softmax", name="predictions"
    )(lstm)
    model = keras.Model(inputs=(input_features, input_words), outputs=outputs)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.9)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def train_model(train_df, devel_df, fasttext, blocksize=128):
    scaler = make_scaler(train_df)
    features = make_features(train_df, scaler)
    words = make_words(train_df, fasttext)
    targets = make_targets(train_df)

    lookup = tf.keras.layers.StringLookup()
    lookup.adapt(targets)

    xf_train_blocks = make_blocks(features, "float32", blocksize)
    xw_train_blocks = make_blocks(words, "float32", blocksize)
    y_train_blocks = make_blocks(lookup(targets), blocksize=blocksize)

    xf_devel_blocks = make_blocks(make_features(devel_df, scaler), "float32", blocksize)
    xw_devel_blocks = make_blocks(make_words(devel_df, fasttext), "float32", blocksize)
    y_devel_blocks = make_blocks(lookup(make_targets(devel_df)), blocksize=blocksize)
    callback = tf.keras.callbacks.EarlyStopping(
        start_from_epoch=25,
        monitor="val_sparse_categorical_accuracy",
        patience=25,
        restore_best_weights=True,
    )
    model = make_model(features.shape[1], lookup.vocabulary_size(), words.shape[1])
    model.fit(
        (xf_train_blocks, xw_train_blocks),
        y_train_blocks,
        batch_size=16,
        epochs=1000,
        validation_data=((xf_devel_blocks, xw_devel_blocks), y_devel_blocks),
        callbacks=(callback,),
    )
    model.evaluate((xf_devel_blocks, xw_devel_blocks), y_devel_blocks)
    return model, scaler, lookup.get_vocabulary()


def chunk(df, model, scaler, vocab, fasttext):
    xf_blocks = make_blocks(make_features(df, scaler), "float32")
    xw_blocks = make_blocks(make_words(df, fasttext), "float32")
    predictions = [
        vocab[i] for i in np.concatenate(model.predict((xf_blocks, xw_blocks), verbose=0).argmax(axis=2))
    ]
    words = df.loc[:, "text"]
    sentences = []
    chunk = []
    tag = None
    for label, word in zip(predictions, words):
        if label[0] == "B":
            if chunk:
                sentences.append((tag, " ".join(str(x) for x in chunk)))
                chunk = []
            tag = label.partition("-")[2]
        if label != "O":
            chunk.append(word)
    sentences.append((tag, " ".join(str(x) for x in chunk)))
    return sentences


def tag(df, model, scaler, vocab, fasttext):
    xf_blocks = make_blocks(make_features(df, scaler), "float32")
    xw_blocks = make_blocks(make_words(df, fasttext), "float32")
    predictions = [
        vocab[i] for i in np.concatenate(model.predict((xf_blocks, xw_blocks), verbose=0).argmax(axis=2))
    ]
    return predictions[: df.shape[0]]


def cmd_train(args):
    devel_df = load_csv(args.csv[-1])
    df = None
    for path in args.csv[:-1]:
        if df is None:
            df = load_csv(path)
        else:
            df = pd.concat([df, load_csv(path)])
    wordvecs = fasttext.load_model(str(MODELPATH.with_suffix(".fasttext")))
    model, scaler, vocab = train_model(df, devel_df, wordvecs)
    if args.output:
        model.save(args.output)
        with open(args.output.with_suffix(".scaler.pkl"), "wb") as outfh:
            pickle.dump(scaler, outfh)
        with open(args.output.with_suffix(".vocab.json"), "wt") as outfh:
            json.dump(vocab, outfh)


def cmd_tag(args):
    model, scaler, vocab, fasttext = load_model(args.model)
    df = load_csv(args.csv)
    df = df.assign(tag=tag(df, model, scaler, vocab, fasttext))
    df.to_csv(sys.stdout, index=False)


def cmd_chunk(args):
    model, scaler, vocab, fasttext = load_model(args.model)
    df = load_csv(args.csv)
    for tag, bloc in chunk(df, model, scaler, vocab, fasttext):
        print(f"{tag}:\n{bloc}\n")


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train", help="entraîner le modèle à partir de fichiers CSV")
    train_parser.add_argument("csv", help="Fichiers CSV d'entree (le dernier servira à la validation)", nargs="+", type=Path)
    train_parser.add_argument("-o", "--output", help="Nom de base pour fichiers de sortie", type=Path)
    train_parser.set_defaults(func=cmd_train)
    tag_parser = subparsers.add_parser("tag", help="générer des étiquettes (tags) pour chaque mot d'un CSV")
    tag_parser.add_argument("csv", help="Fichier CSV d'entree", type=Path)
    tag_parser.add_argument("-m", "--model", help="Nom de base pour modèle", type=Path)
    tag_parser.set_defaults(func=cmd_tag)
    chunk_parser = subparsers.add_parser("chunk", help="analyzer un CSV en blocs de texte étiquettés")
    chunk_parser.add_argument("csv", help="Fichier CSV d'entree", type=Path)
    chunk_parser.add_argument("-m", "--model", help="Nom de base pour modèle", type=Path)
    chunk_parser.set_defaults(func=cmd_chunk)
    return parser


if __name__ == "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    args.func(args)
