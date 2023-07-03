import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INDEXLIKE = re.compile(r"^(\d+|[a-z])[\)\.]|[•-]$")


def make_feats_from_csv(path: Path, vocab: dict[str], transform=None, target_transform=None, predict=False):
    # Load data frame
    df = pd.read_csv(path)
    # Split into pages
    for idx, page in df.groupby("page"):
        yield make_feats(page, vocab, transform, target_transform, predict)


def make_feats(df: pd.DataFrame, vocab: dict[str], transform=None, target_transform=None, predict=False) -> pd.DataFrame:
    df = df.assign(
        xdelta=df.x0.diff(),
        ydelta=df.doctop.diff(),
        height=df.bottom - df.top,
        width=df.x1 - df.x0,
    )
    df = df.assign(xdd=df.xdelta.diff(), ydd=df.ydelta.diff())
    df = df.fillna(0)
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
    feats = feats.assign(
        x0=df.loc[:, "x0"], width=df.loc[:, "width"], height=df.loc[:, "height"]
    )
    feats = feats.assign(
        index=index,
        idelta=idelta,
        isnum=isnum,
        iseq=iseq,
    )
    tags = df.loc[:, "tag"]
    if transform:
        feats = feats.apply(transform)
    if target_transform:
        tags = tags.apply(target_transform)
    if predict:  # Do not add OOVs
        tags = tags.apply(lambda t: vocab.get(t, 0))
    else:
        tags = tags.apply(lambda t: vocab.setdefault(t, len(vocab)))
    return (torch.from_numpy(feats.to_numpy(dtype=np.float32)),
            torch.from_numpy(tags.to_numpy(dtype=np.int64)))


def simplify_tags(tag):
    """Enlever certains tags difficiles a predire"""
    if 'Tableau' in tag:
        return 'O'  # They just do not work, have to do them elsewhere
    if 'TOC' in tag:
        return 'O'  # Begone
    elif tag[0] == 'I':  # For RNNs (without CRF output) this is better
        return tag.partition('-')[0]
    elif tag == 'B-Amendement':  # Impossible pour le moment sans couleur
        return 'B-Alinea'
    elif tag == 'B-SousSection':  # Peut faire en post-processing
        return 'B-Section'
    elif tag == 'B-Annexe':  # Idem
        return 'B-Chapitre'
    return tag


class PageDataset(Dataset):
    def __init__(self, csvfiles: list[Path], transform=None, target_transform=None, vocab=None, predict=False):
        self.pages = []
        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            self.vocab = {"[UNK]": 0}
        for path in csvfiles:
            self.pages.extend(make_feats_from_csv(path, self.vocab, transform, target_transform, predict))

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, idx: int):
        return self.pages[idx]

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
