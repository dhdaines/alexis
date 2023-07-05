"""
Fonction de conversion de PDF en CSV d'ALEXI.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Iterator

import pdfplumber
from tqdm import tqdm

LOGGER = logging.getLogger("pdf2csv")


def detect_margins(page, words) -> Iterator[dict[str, Any]]:
    if not words:
        return words
    margin_top = 0
    margin_bottom = page.height
    l1top = words[0]["top"]
    l1bottom = words[0]["bottom"]
    l1size = l1bottom - l1top
    l1 = [word["text"] for word in words if word["top"] == l1top]
    letop = words[-1]["top"]
    lebottom = words[-1]["bottom"]
    lesize = lebottom - letop
    le = [word["text"] for word in words if word["top"] == letop]
    if len(le) == 1 and le[0].isnumeric() and len(le[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en pied trouvé à %f pt", page.page_number, letop
        )
        margin_bottom = letop
    elif lesize < 10 and (page.height - lebottom) < 72:
        LOGGER.info("page %d: pied de page trouvé à %f pt", page.page_number, letop)
        margin_bottom = letop
        # Il existe parfois deux lignes de pied de page
        for w in words[::-1]:
            if w["top"] == letop:
                continue
            wsize = w["bottom"] - w["top"]
            if wsize < 10 and letop - w["bottom"] < 10:
                LOGGER.info(
                    "page %d: deuxième ligne de pied de page trouvé à %f pt",
                    page.page_number,
                    w["top"],
                )
                margin_bottom = w["top"]
                break
    if len(l1) == 1 and l1[0].isnumeric() and len(l1[0]) < 4:
        LOGGER.info(
            "page %d: numéro de page en tête trouvé à %f", page.page_number, l1bottom
        )
        margin_top = l1bottom
    elif l1size < 10 and l1top < 72:
        LOGGER.info("page %d: en-tête trouvé a %f pt", page.page_number, l1bottom)
        margin_top = l1bottom
    seen_head = seen_foot = False
    for word in words:
        if word["bottom"] <= margin_top:
            word["tag"] = "I-Tete" if seen_head else "B-Tete"
            seen_head = True
        elif word["top"] >= margin_bottom:
            word["tag"] = "I-Pied" if seen_foot else "B-Pied"
            seen_foot = True
        yield word


def write_csv(pdf: pdfplumber.PDF, path: Path):
    fields = []
    for p in pdf.pages:
        words = p.extract_words()
        if words:
            fields = list(words[0].keys())
            break
    if not fields:
        return
    with open(path, "wt") as ofh:
        fields.remove("text")
        fieldnames = [
            "tag",
            "text",
            "page",
            "page_width",
            "page_height",
            "r",
            "g",
            "b",
        ] + fields
        writer = csv.DictWriter(ofh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(tqdm(pdf.pages)):
            words = p.extract_words()
            words = detect_margins(p, words)
            # Index characters for lookup
            chars = dict(((c["x0"], c["top"]), c) for c in p.chars)
            LOGGER.info("traitement de la page %d", p.page_number)
            for w in words:
                # Extract colour from first character (FIXME: assumes RGB space)
                if c := chars.get((w["x0"], w["top"])):
                    # OMG WTF pdfplumber!!!
                    if isinstance(c["stroking_color"], list) or isinstance(
                        c["stroking_color"], tuple
                    ):
                        if len(c["stroking_color"]) == 1:
                            w["r"] = w["g"] = w["b"] = c["stroking_color"][0]
                        elif len(c["stroking_color"]) == 3:
                            w["r"], w["g"], w["b"] = c["stroking_color"]
                        else:
                            LOGGER.warning(
                                "Espace couleur non pris en charge: %s",
                                c["stroking_color"],
                            )
                    else:
                        w["r"] = w["g"] = w["b"] = c["stroking_color"]
                w["page"] = p.page_number
                w["page_height"] = round(float(p.height))
                w["page_width"] = round(float(p.width))
                # Round positions to points
                for field in "x0", "x1", "top", "bottom", "doctop":
                    w[field] = round(float(w[field]))
                writer.writerow(w)


def main(args):
    """Convertir les PDF en CSV"""
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.verbose:
        global tqdm
        tqdm = lambda x: x  # noqa: E731
    with pdfplumber.open(args.infile) as pdf:
        logging.info("processing %s", args.infile)
        write_csv(pdf, args.outfile)
