"""Extraction de segments de textes avec CRF"""

import csv
import itertools
import re
from pathlib import Path
from typing import Iterable, Any

import joblib

from alexi.analyse import group_iob
from alexi.convert import FIELDNAMES, Converteur
from alexi.segment import Segmenteur, page2features, page2labels, split_pages

FEATNAMES = [name for name in FIELDNAMES if name != "seqtag"]
DEFAULT_MODEL = Path(__file__).parent / "models" / "crfseq.joblib.gz"
NUMDASH = re.compile(r"[\d-]+")


def features(_, word):
    features = ["bias"]
    features.append("lower=%s" % word["text"].lower())
    features.append("reg=%s" % bool(word["text"].lower() == "règlement"))
    features.append("n=%s" % bool(word["text"][0] in "Nn"))
    features.append("alpha=%s" % bool(word["text"].isalpha()))
    features.append("numdash=%s" % bool(NUMDASH.match(word["text"])))
    features.append("bold=%s" % bool("bold" in word["fontname"].lower()))
    features.append("tag=%s" % word["segtag"].partition("-")[2])
    features.append("size=%d" % (int(word["bottom"]) - int(word["top"])))
    return features


def labels(_, word):
    return word.get("seqtag", "O")


def load(paths):
    for p in paths:
        with open(Path(p), "rt") as infh:
            reader = csv.DictReader(infh)
            last_page = None
            for page in split_pages(reader):
                if last_page is None:
                    yield page
                last_page = page
            yield last_page


def make_data(dataset, n=2):
    return zip(
        *((page2features(s, features, n), page2labels(s, labels)) for s in dataset)
    )


class Extracteur:
    def __init__(self, model=DEFAULT_MODEL):
        self.crf = joblib.load(model)

    def predict(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        first_page = last_page = None
        for page in split_pages(words):
            if last_page is None:
                first_page = page
            else:
                if last_page is first_page:
                    yield from self.crf.predict_single(
                        page2features(last_page, feature_func=features, n=2)
                    )
                else:
                    for _ in last_page:
                        yield "O"
            last_page = page
        if last_page and last_page is not first_page:
            yield from self.crf.predict_single(
                page2features(last_page, feature_func=features, n=2)
            )

    def __call__(self, words: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        c1, c2 = itertools.tee(words)
        pred = self.predict(c1)
        for label, word in zip(pred, c2):
            word["seqtag"] = label
            yield word


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfpath", type=Path, help="Fichier PDF")
    args = parser.parse_args()
    conv = Converteur(args.pdfpath)
    seg = Segmenteur()
    ex = Extracteur()
    pages = conv.extract_words()
    segmented = seg(pages)
    tagged = ex(segmented)
    for bloc in group_iob(tagged, "seqtag"):
        print(f"{bloc.type}: {bloc.texte}")


if __name__ == "__main__":
    main()
