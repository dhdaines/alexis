"""Extraire le contenu textuel des HTML et PDF de lois et règlements
référenciés de LegisQuébec.
"""

import argparse
import itertools
import json
import logging
from collections import deque

from lxml import html, etree
from pathlib import Path

from alexi.convert import Converteur, write_csv

LOGGER = logging.getLogger(Path(__file__).stem)


def linearise(tree):
    """Lineariser XHTML pour alignement."""
    q = deque(tree.getroot())
    stacks = {}
    text = []

    def add_text(stack, t):
        stack_id = stacks.setdefault(",".join(stack), len(stacks))
        LOGGER.debug("add_text %d: %s", stack_id, t)
        text.append((stack_id, t))

    gen_id = itertools.count()
    stack = []
    while q:
        el = q.popleft()
        if el is None:
            # No tail
            stack.pop()
            continue
        elif isinstance(el, str):
            # Tail belongs to enclosing element
            stack.pop()
            add_text(stack, el)
            continue
        el_id = el.get("id", "__" + str(next(gen_id)))
        el_ctx = "|".join((el.tag, el_id, el.get("class", "")))
        stack.append(el_ctx)
        LOGGER.debug("Element %s", el_ctx)
        # block elements (there is only div) get surrounded in newlines
        if el.tag == "div":
            # Make sure to add the \n even if there is no leading text
            # (the div starts with a span for instance)
            el.text = "\n" + ("" if el.text is None else el.text)
        if el.text:
            add_text(stack, el.text)
        if el.tag == "div":
            q.appendleft("\n" if el.tail is None else el.tail + "\n")
        else:
            q.appendleft(el.tail)
        q.extendleft(reversed(el))
    # invert the stacks to get a list of contexts
    ctx = []
    for pile, idx in stacks.items():
        while idx >= len(ctx):
            ctx.append("")
        ctx[idx] = pile
    return ctx, text


def extract(path: Path):
    """Extraire le contenu des lois"""
    xhtml_path = path.with_suffix(".xhtml")
    json_path = path.with_suffix(".json")
    LOGGER.info("Extracting %s to %s and %s", path, xhtml_path, json_path)
    with open(path, "rb") as infh:
        tree = html.parse(infh)
        xmldiv = tree.find(".//div[@xmlns='http://www.w3.org/1999/xhtml']")
        if xmldiv is None:
            LOGGER.warning("No XML found in %s", path)
            return
        # Remove the @!#$!@#$! default namespace
        del xmldiv.attrib["xmlns"]
        xmldiv.tail = ""
        tree._setroot(xmldiv)
        with open(xhtml_path, "wb") as outfh:
            outfh.write(b'<?xml-stylesheet type="text/css" href="style.css"?>\n')
            outfh.write(
                etree.tostring(
                    tree,
                    method="xml",
                    encoding="utf-8",
                )
            )
        with open(json_path, "wt") as outfh:
            ctx, text = linearise(tree)
            json.dump({"ctx": ctx, "text": text}, outfh, indent=2, ensure_ascii=False)
    pdfpath = path.with_suffix(".pdf")
    csvpath = path.with_suffix(".csv")
    LOGGER.info("Extracting %s to %s", pdfpath, csvpath)
    conv = Converteur(pdfpath)
    with open(csvpath, "wt") as outfh:
        write_csv(conv.extract_words(), outfh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("htmls", help="Fichiers HTMLS", type=Path, nargs="+")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for path in args.htmls:
        extract(path)


if __name__ == "__main__":
    main()
