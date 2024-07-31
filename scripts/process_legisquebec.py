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
    prev_ctx = ""
    prev_tag = "O"

    def enclosing_block(ctx):
        stack = ctx.split(",")
        for el in reversed(stack):
            tag, _, _ = el.split("|")
            if tag == "div":
                return el

    for w, c, x in zip(iobs, *alignment):
        if x == "\x00":
            w["segment"] = "O"
            prev_tag = "O"
            prev_block = None
            continue
        elif c == "\x00":
            xw, ctx = next(xitor)
            LOGGER.warning("Skipping word in XHTML: %s", xw)
            continue

        xw, contexts = next(xitor)
        cid = contexts[0]
        ctx = xhtml["ctx"][cid]
        block = enclosing_block(ctx)
        LOGGER.debug(
            "CSV word %s XHTML word %s cid %d block %s context %s",
            c,
            x,
            cid,
            block,
            ctx,
        )
        if "Label-Section" in ctx:
            tag = "Article"
        elif "group5" in ctx:
            tag = "Section"
        elif "Paragraph" in ctx:
            tag = "Liste"
        else:
            # HistoricalNote is Alinea, but we can set sequence=Amendement...
            tag = "Alinea"

        if tag != prev_tag:
            iob = "B"
        else:
            # Some particular rules
            if "Heading" in ctx or "HistoricalNote" in ctx:
                # Heading and HistoricalNote make up a single element
                # (there are sequence tags within them...)
                iob = "I"
            else:
                # Alinea, Liste, will break on enclosing blocks
                iob = "I" if block == prev_block else "B"
        w["segment"] = f"{iob}-{tag}"
        prev_block = block
        prev_ctx = ctx
        prev_tag = tag
    convert.write_csv(iobs, sys.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("path", help="Fichier CSV", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    process(args.path, args.path.with_suffix(".json"))


if __name__ == "__main__":
    main()
