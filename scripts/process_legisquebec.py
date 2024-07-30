"""Analyser le XML des lois et règlements référenciés de LegisQuébec
par référence aux PDF pour créer des données d'entraînement.
"""

import argparse
import itertools
import json
import logging
import re

from pathlib import Path
from sequence_align.pairwise import hirschberg

from alexi import segment

LOGGER = logging.getLogger(Path(__file__).stem)


def process(csvpath, jsonpath):
    """Align words in CSV with words extracted from XML.

    We know that the CSV will contain extra stuff (table of contents,
    headers, footers) but that it is (more or less)
    whitespace-normalized, so it should align closely to the XML.

    The XML will have extra whitespace to delimit block-level
    elements, and conversely some words will be split between mutiple
    inline elements.  So we need to re-tokenize it tracking the
    original context IDs.

    """
    iobs = segment.load([csvpath])
    cwords = [w["text"] for w in iobs]
    with open(jsonpath, "rt") as infh:
        xhtml = json.load(infh)
    xctx = []
    xwords = []
    # Join text and track positions.  We skip hidden text here since
    # it certainly won't align with the PDF (FIXME: there is some
    # hidden text in the PDF too which we need to figure out how to
    # remove)
    visible = [
        (ctx, txt) for (ctx, txt) in xhtml["text"] if "Hidden" not in xhtml["ctx"][ctx]
    ]
    # Retokenize on non-whitespace and accumulate contexts
    xtext = "".join(txt for ctx, txt in visible)
    xpos = [0, *itertools.accumulate(len(txt) for ctx, txt in visible)]
    itor = itertools.pairwise(zip((ctx for ctx, txt in visible), xpos))
    (ctx, pos), (_, next_pos) = next(itor)
    for m in re.finditer(r"\S+", xtext):
        start, end = m.span()
        xwords.append(m[0])
        print(m[0], start, end)
        print(ctx, f"{pos}:{next_pos}")
        while start >= next_pos:
            (ctx, pos), (next_ctx, next_pos) = next(itor)
            print("->", ctx, f"{pos}:{next_pos}")
        tctx = [ctx]
        while end >= next_pos:
            (ctx, pos), (next_ctx, next_pos) = next(itor)
            if next_pos > end:
                break
            tctx.append(ctx)
            print("+>", ctx, f"{pos}:{next_pos}")
        print(" =>", tctx)
        print()
        xctx.append(tctx)
    alignment = hirschberg(cwords, xwords, gap="\x00")
    print("ALIGN", len(alignment[0]), len(cwords), len(xwords))
    xitor = zip(xwords, xctx)
    for c, x in zip(*alignment):
        if x != "\x00":
            xw, ctx = next(xitor)
            print(c, xw, ctx)
        else:
            print(c, "O")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Fichier CSV", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    process(args.path, args.path.with_suffix(".json"))


if __name__ == "__main__":
    main()
