"""Analyser le XML des lois et règlements référenciés de LegisQuébec
par référence aux PDF pour créer des données d'entraînement.
"""

import argparse
import itertools
import json
import logging
import re
import sys

from pathlib import Path
from sequence_align.pairwise import hirschberg

from alexi import segment, convert

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
    iobs = list(segment.load([csvpath]))
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
        LOGGER.debug("token %s %d:%d", m[0], start, end)
        LOGGER.debug("ctx %s %d:%d", ctx, pos, next_pos)
        while start >= next_pos:
            (ctx, pos), (_, next_pos) = next(itor)
            LOGGER.debug("-> %s %d:%d", ctx, pos, next_pos)
        tctx = [ctx]
        while end >= next_pos:
            (ctx, pos), (_, next_pos) = next(itor)
            if next_pos > end:
                break
            tctx.append(ctx)
            LOGGER.debug("+> %s %d:%d", ctx, pos, next_pos)
        LOGGER.debug(" => %s", tctx)
        xctx.append(tctx)
    alignment = hirschberg(cwords, xwords, gap="\x00")
    LOGGER.debug("ALIGN %d %d %d", len(alignment[0]), len(cwords), len(xwords))
    xitor = zip(xwords, xctx)
    prev_ctx = -1
    for w, c, x in zip(iobs, *alignment):
        if x == "\x00":
            w["segment"] = "O"
            prev_ctx = -1
        elif c.lower() != x.lower():
            w["segment"] = "O"
            prev_ctx = -1
            xw, ctx = next(xitor)
        else:
            xw, ctx = next(xitor)
            if ctx != prev_ctx:  # FIXME: Not quite right
                iob = "B"
            else:
                iob = "I"
            ctxtxt = ";".join(xhtml["ctx"][cc] for cc in ctx)
            if "Label-Section" in ctxtxt:
                tag = "Article"
            elif "group5" in ctxtxt:
                tag = "Section"
            elif "Paragraph" in ctxtxt:
                tag = "Liste"
            else:
                tag = "Alinea"

            w["segment"] = f"{iob}-{tag}"
        prev_ctx = ctx
    convert.write_csv(iobs, sys.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Fichier CSV", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    process(args.path, args.path.with_suffix(".json"))


if __name__ == "__main__":
    main()
