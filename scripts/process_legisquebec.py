"""Analyser le XML des lois et règlements référenciés de LegisQuébec
par référence aux PDF pour créer des données d'entraînement.
"""

import argparse
import itertools
import json
import logging
import re
import sys

import more_itertools
from pathlib import Path
from sequence_align.pairwise import hirschberg

from alexi import segment, convert

LOGGER = logging.getLogger(Path(__file__).stem)
GAP = "\x00"


def align(cwords, xwords):
    """Align words in CSV with words extracted from XML.

    We know that the CSV will contain extra stuff (table of contents,
    headers, footers) but that it is (more or less)
    whitespace-normalized, so it should align closely to the XML.  We
    will eliminate some obvious artifacts in order to facilitate this
    alignment.

    The XML will have extra whitespace to delimit block-level
    elements, and conversely some words will be split between mutiple
    inline elements.  So we need to re-tokenize it tracking the
    original context IDs.
    """
    # Find extents of table of contents
    start_toc = start_doc = -1
    for start_toc, triple in enumerate(more_itertools.triplewise(cwords)):
        threewords = " ".join(triple).lower().replace("è", "e")
        LOGGER.debug("%d %r -> %s", start_toc, triple, threewords)
        if threewords == "table des matieres":
            LOGGER.debug("Found TOC at position %d", start_toc)
            break
    else:
        start_toc = -1
    # Take first entry in table of contents
    # Scan ahead until we find it again, this is the start of the document itself
    if start_toc != -1:
        first_pos = start_toc + 3
        first_ent = " ".join(cwords[first_pos : first_pos + 2]).lower()
        LOGGER.debug("First entry in TOC: %s", first_ent)
        for start_doc, pair in enumerate(itertools.pairwise(cwords[first_pos + 2 :])):
            twowords = " ".join(pair).lower()
            LOGGER.debug("%d %r -> %s", start_doc, pair, twowords)
            if twowords == first_ent:
                start_doc += first_pos + 2
                LOGGER.debug("Found document at position %d", start_doc)
                break
        else:
            start_doc = -1
    if start_toc != -1 and start_doc != -1:
        LOGGER.debug("Excluding TOC from %d to %d. Alignment:", start_toc, start_doc)
        alignment = hirschberg(cwords[:start_toc] + cwords[start_doc:], xwords, gap=GAP)
        # Track insertions/deletions to find TOC location in alignment for reinsertion
        cpos = xpos = 0
        gap_pos = -1
        for gap_pos, (cw, xw) in enumerate(zip(*alignment)):
            LOGGER.debug(
                "%d:%s=%s %d:%s=%s", cpos, cwords[cpos], cw, xpos, xwords[xpos], xw
            )
            if cpos == start_toc:
                LOGGER.debug("Reinserting gap at TOC at position %d", gap_pos)
                break
            if cw != GAP:
                cpos += 1
            if xw != GAP:
                xpos += 1
        assert gap_pos != -1
        gap = [GAP] * (start_doc - start_toc)
        return (
            alignment[0][:gap_pos]
            + cwords[start_toc:start_doc]
            + alignment[0][gap_pos:],
            alignment[1][:gap_pos] + gap + alignment[1][gap_pos:],
        )
    else:
        return hirschberg(cwords, xwords, gap=GAP)


def process(csvpath, jsonpath):
    """Assign segment tags to PDF using XML alignment."""
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

    alignment = align(cwords, xwords)
    LOGGER.debug("ALIGN %d %d %d", len(alignment[0]), len(cwords), len(xwords))
    xitor = zip(xwords, xctx)
    prev_tag = "O"
    prev_block = ""

    def enclosing_block(ctx):
        stack = ctx.split(",")
        for el in reversed(stack):
            tag, _, _ = el.split("|")
            if tag == "div":
                return el

    for w, c, x in zip(iobs, *alignment):
        if x == GAP:
            w["segment"] = "O"
            prev_tag = "O"
            prev_block = None
            continue
        elif c == GAP:
            xw, ctx = next(xitor)
            LOGGER.warning("Skipping word in XHTML: %s", xw)
            continue

        xw, contexts = next(xitor)
        cid = contexts[0]
        ctx = xhtml["ctx"][cid]
        block = enclosing_block(ctx)
        LOGGER.debug(
            "CSV word %s XHTML word %s cid %d context %s",
            c,
            x,
            cid,
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

        LOGGER.debug(
            "tag %s prev_tag %s block %s prev_block %s",
            tag,
            prev_tag,
            block,
            prev_block,
        )
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
