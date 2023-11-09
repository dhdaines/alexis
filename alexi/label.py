"""Extraction de segments de textes avec CRF"""

import itertools
import csv
import re
from pathlib import Path

from alexi.analyse import group_iob
from alexi.convert import FIELDNAMES, Converteur
from alexi.segment import Segmenteur, page2features, page2labels, split_pages

FEATNAMES = [name for name in FIELDNAMES if name != "seqtag"]
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


def tag_sequence(crf, pages, n=2):
    page_list = list(pages)
    X = [page2features(s, features, n) for s in page_list]
    y = crf.predict(X)
    flat_y = itertools.chain.from_iterable(y)
    for w, label in zip(itertools.chain.from_iterable(page_list), flat_y):
        w["seqtag"] = label
        yield w


def extract_pdf(pdfpath):
    conv = Converteur(pdfpath)
    seg = Segmenteur()
    pages = conv.extract_words((0, len(conv.pdf.pages) - 1))
    segmented = split_pages(seg(pages))
    tagged = tag_sequence(segmented)
    for bloc in group_iob(tagged, "seqtag"):
        print(f"{bloc.type}: {bloc.texte}")
