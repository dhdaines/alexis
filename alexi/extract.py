"""
Convertir les règlements en HTML
"""

import argparse
import csv
import itertools
import json
import logging
import operator
import os
from collections import deque
from pathlib import Path
from typing import Any, Iterable, TextIO

from alexi.analyse import Analyseur, Bloc, Document, Element
from alexi.analyse import extract_zonage, extract_links
from alexi.convert import Converteur
from alexi.format import format_html
from alexi.label import DEFAULT_MODEL as DEFAULT_LABEL_MODEL
from alexi.label import Extracteur
from alexi.segment import DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL
from alexi.segment import DEFAULT_MODEL_NOSTRUCT, Segmenteur

LOGGER = logging.getLogger("extract")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments to the argparse"""
    parser.add_argument(
        "-o", "--outdir", help="Repertoire de sortie", type=Path, default="export"
    )
    parser.add_argument(
        "-n", "--no-images", help="Ne pas extraire les images", action="store_true"
    )
    parser.add_argument(
        "-C",
        "--no-csv",
        help="Ne pas utiliser le CSV de référence",
        action="store_true",
    )
    parser.add_argument("--segment-model", help="Modele CRF", type=Path)
    parser.add_argument(
        "--label-model", help="Modele CRF", type=Path, default=DEFAULT_LABEL_MODEL
    )
    parser.add_argument(
        "-m",
        "--metadata",
        help="Fichier JSON avec metadonnées des documents",
        type=Path,
    )
    parser.add_argument(
        "docs", help="Documents en PDF ou CSV pré-annoté", type=Path, nargs="+"
    )
    return parser


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "rt") as infh:
        return list(csv.DictReader(infh))


HTML_GLOBAL_HEADER = """<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css" integrity="sha384-X38yfunGUhNzHpBaEBsWLO+A0HDYOQi8ufWDkZ0k9e0eXz/tH3II7uKZ9msv++Ls" crossorigin="anonymous">
"""
STYLE_CSS = """html, body {
    margin: 0;
    height: 100%;
}
.container {
    display: flex;
    flex-flow: column;
    height: 100%;
}
#header {
    font-family: sans-serif;
    text-align: center;
    text-transform: uppercase;
    padding: 0.5ex;
    margin: 0;
    background: #2d3e50;
    color: #eee;
}
.initial {
    color: #aaa;
}
#body {
    overflow-y: scroll;
    padding: 2px;
}
@media (max-width: 599px) {
    .nomobile {
        display: none;
    }
}
@media (min-width: 600px) {
    #body {
        padding: 20px;
    }
}
ul {
    padding-left: 1em;
}
li {
    list-style-type: none;
}
li.text {
    margin-left: -1em;
    margin-top: 0.5em;
    margin-bottom: 1em;
}
details {
    margin-bottom: 1em;
}
summary {
    cursor: pointer;
}
li.leaf {
    list-style-type: disc;
    margin-bottom: 0.25em;
}
"""


def extract_element(
    doc: Document,
    el: Element,
    outdir: Path,
    imgdir: Path,
):
    """Extract the various constituents, referencing images in the
    generated image directory."""
    # Can't use Path.relative_to until 3.12 :(
    rel_imgdir = os.path.relpath(imgdir, outdir)
    rel_style = os.path.relpath(imgdir.parent.parent / "style.css", outdir)
    doc_titre = el.titre
    if doc.titre != "Document":
        doc_titre = doc.titre
        if doc.numero:
            doc_titre = f'{doc.numero} <span class="nomobile">{doc.titre}</span>'
    HTML_HEADER = (
        HTML_GLOBAL_HEADER
        + f"""    <link rel="stylesheet" href="{rel_style}">
    <title>{el.titre}</title>
  </head>
  <body>
    <div class="container">
    <h1 id="header">{doc_titre}</h1>
    <div id="body">
"""
    )
    HTML_FOOTER = """</div></div></body>
</html>
"""
    outdir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("%s %s", outdir, el.titre)
    with open(outdir / "index.html", "wt") as outfh:
        outfh.write(HTML_HEADER)
        outfh.write(format_html(doc, element=el, imgdir=rel_imgdir, fragment=True))
        outfh.write(HTML_FOOTER)


def make_index_html(
    topdir: Path, docdir: Path, title: str, elements: Iterable[Element]
):
    """Create an index.html for docdir."""
    style = os.path.relpath(topdir / "style.css", docdir)
    HTML_HEADER = (
        HTML_GLOBAL_HEADER
        + f"""    <link rel="stylesheet" href="{style}">
    <title>{title}</title>
  </head>
  <body>
    <div class="container">
    <h1 id="header">{title}</h1>
    <ul id="body">
"""
    )
    lines = []
    off = "    "
    sp = "  "
    for el in elements:
        lines.append(f"{off}{sp}<li>")
        if el.numero[0] == "_":
            titre = el.titre if el.titre else f"{el.type} (numéro inconnu)"
        else:
            lines.append(f'{off}{sp}{sp}<span class="number">{el.numero}</span>')
            titre = el.titre if el.titre else f"{el.type} {el.numero}"
        lines.append(
            f'{off}{sp}{sp}<a href="{el.numero}/index.html" class="title">{titre}</a>'
        )
        lines.append(f"{off}{sp}</li>")
    HTML_FOOTER = """</ul>
    </div>
  </body>
</html>
"""
    docdir.mkdir(parents=True, exist_ok=True)
    with open(docdir / "index.html", "wt") as outfh:
        LOGGER.info("Génération de %s", docdir / "index.html")
        outfh.write(HTML_HEADER)
        for line in lines:
            print(line, file=outfh)
        outfh.write(HTML_FOOTER)


def save_images_from_pdf(blocs: list[Bloc], conv: Converteur, docdir: Path):
    """Convertir des éléments du PDF difficiles à réaliser en texte en
    images et les insérer dans la structure du document (liste de blocs).
    """
    images: dict[int, list[Bloc]] = {}
    for bloc in blocs:
        if bloc.type in ("Tableau", "Figure"):
            assert isinstance(bloc.page_number, int)
            images.setdefault(bloc.page_number, []).append(bloc)
    for page_number, image_blocs in images.items():
        page = conv.pdf.pages[page_number - 1]
        for bloc in image_blocs:
            x0, top, x1, bottom = bloc.bbox
            if x0 == x1 or top == bottom:
                LOGGER.warning("Skipping empty image bbox %s", bloc.bbox)
                continue
            x0 = max(0, x0)
            top = max(0, top)
            x1 = min(page.width, x1)
            bottom = min(page.height, bottom)
            img = page.crop((x0, top, x1, bottom)).to_image(
                resolution=150, antialias=True
            )
            LOGGER.info("Extraction de %s", docdir / bloc.img)
            img.save(docdir / bloc.img)


def make_redirect(path: Path, target: Path):
    """Creer une redirection HTML."""
    path.mkdir(exist_ok=True)
    with open(path / "index.html", "wt") as outfh:
        outfh.write(
            f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta http-equiv="refresh" content="0;URL='../{target}/" />
    <title></title>
</head>
<body>
    <p>La version actuelle du règlement {path.name} se trouve à {target}.</p>
</body>
</html>
"""
        )


def extract_html(args, path, iob, conv):
    docdir = args.outdir / path.stem
    imgdir = args.outdir / path.stem / "img"
    LOGGER.info("Génération de pages HTML sous %s", docdir)
    docdir.mkdir(parents=True, exist_ok=True)
    analyseur = Analyseur(path.stem, iob)
    if conv and not args.no_images:
        LOGGER.info("Extraction d'images sous %s", imgdir)
        imgdir.mkdir(parents=True, exist_ok=True)
        images = conv.extract_images()
        analyseur.add_images(images)
        save_images_from_pdf(analyseur.blocs, conv, imgdir)
    LOGGER.info("Analyse de la structure de %s", path)
    return analyseur()


def output_html(args, path, doc):
    docdir = args.outdir / path.stem
    imgdir = args.outdir / path.stem / "img"
    # Do articles/annexes at top level
    seen_paliers = set()
    doc_titre = doc.titre if doc.titre != "Document" else path.stem
    for palier in ("Article", "Annexe"):
        if palier not in doc.paliers:
            continue
        seen_paliers.add(palier)
        if not doc.paliers[palier]:
            continue
        # These go in the top level
        for idx, el in enumerate(doc.paliers[palier]):
            extract_element(doc, el, docdir / palier / el.numero, imgdir)
        make_index_html(
            args.outdir, docdir / palier, f"{doc_titre}: {palier}s", doc.paliers[palier]
        )

    # Now do the rest of the Document hierarchy if it exists
    def make_sub_index(el: Element, path: Path, titre: str):
        subtypes = list(el.sub)
        subtypes.sort(key=operator.attrgetter("type"))
        for subtype, elements in itertools.groupby(
            subtypes, operator.attrgetter("type")
        ):
            if subtype not in seen_paliers:
                make_index_html(
                    args.outdir, path / subtype, f"{titre}: {subtype}s", elements
                )

    top = Path(docdir)
    # Create index.html for immediate descendants (Chapitre, Article, Annexe, etc)
    if doc.structure.sub:
        make_sub_index(doc.structure, docdir, doc_titre)
    # Extract content and create index.html for descendants of all elements
    d = deque((el, top) for el in doc.structure.sub)
    while d:
        el, parent = d.popleft()
        if el.type in seen_paliers:
            continue
        extract_element(doc, el, parent / el.type / el.numero, imgdir)
        if not el.sub:
            continue
        make_sub_index(el, parent / el.type / el.numero, f"{el.type} {el.numero}")
        d.extendleft(
            (subel, parent / el.type / el.numero) for subel in reversed(el.sub)
        )
    # And do a full extraction (which might crash your browser)
    extract_element(doc, doc.structure, docdir, imgdir)
    return doc


def make_doc_subtree(doc: Document, outfh: TextIO):
    """
    Générer HTML pour les contenus d'un document.
    """
    outfh.write("<ul>\n")
    outfh.write(
        f'<li class="text"><a target="_blank" href="{doc.fileid}/index.html">Texte intégral</a>\n'
    )
    if doc.pdfurl is not None:
        outfh.write(f'(<a target="_blank" href="{doc.pdfurl}">PDF</a>)')
    outfh.write("</li>\n")
    top = Path(doc.fileid)
    d = deque((el, top, 1) for el in doc.structure.sub)
    prev_level = 1
    while d:
        el, parent, level = d.popleft()
        if el.type in ("Article", "Annexe"):
            eldir = top / el.type / el.numero
        else:
            eldir = parent / el.type / el.numero
        if el.numero[0] == "_":
            if el.titre:
                eltitre = el.titre
            else:
                eltitre = el.type
        else:
            if el.titre:
                eltitre = f"{el.type} {el.numero}: {el.titre}"
            else:
                eltitre = f"{el.type} {el.numero}"
        while level < prev_level:
            outfh.write("</ul></details></li>\n")
            prev_level -= 1
        if el.sub:
            outfh.write(
                f'<li class="{el.type} node"><details><summary>{eltitre}</summary><ul>\n'
            )
            link = f'<a target="_blank" href="{eldir}/index.html">Texte intégral</a>'
            pdflink = ""
            if doc.pdfurl is not None:
                pdflink = (
                    f' (<a target="_blank" href="{doc.pdfurl}#page={el.page}">PDF</a>)'
                )
            outfh.write(f'<li class="text">{link}{pdflink}</li>\n')
        else:
            link = f'<a target="_blank" href="{eldir}/index.html">{eltitre}</a>'
            pdflink = ""
            if doc.pdfurl is not None:
                pdflink = (
                    f' (<a target="_blank" href="{doc.pdfurl}#page={el.page}">PDF</a>)'
                )
            outfh.write(f'<li class="{el.type} leaf">{link}{pdflink}</li>\n')
        d.extendleft((subel, eldir, level + 1) for subel in reversed(el.sub))
        prev_level = level
    while prev_level > 1:
        outfh.write("</ul></details></li>\n")
        prev_level -= 1
    outfh.write("</ul>\n")


def make_doc_tree(docs: list[Document], outdir: Path) -> list[dict]:
    HTML_HEADER = (
        HTML_GLOBAL_HEADER
        + """    <title>ALEXI</title>
    <link rel="stylesheet" href="./style.css">
  </head>
  <body>
    <div class="container">
    <h1 id="header"><span class="initial">AL</span>EXI<span class="nomobile">:
        <span class="initial">EX</span>tracteur
        d’<span class="initial">I</span>nformation
        </span>
    </h1>
    <ul id="body">
"""
    )
    HTML_FOOTER = """</ul>
    </div>
  </body>
</html>
"""
    metadata = {}
    docs.sort(key=operator.attrgetter("numero"))
    with open(outdir / "index.html", "wt") as outfh:
        LOGGER.info("Génération de %s", outdir / "index.html")
        outfh.write(HTML_HEADER)
        for doc in docs:
            outfh.write('<li class="Document node"><details>\n')
            outfh.write(f"<summary>{doc.numero}: {doc.titre}</summary>\n")
            make_doc_subtree(doc, outfh)
            outfh.write("</details></li>\n")
            doc_metadata = {
                "numero": doc.numero,
                "titre": doc.titre,
            }
            if doc.pdfurl is not None:
                doc_metadata["pdf"] = doc.pdfurl
            metadata[doc.fileid] = doc_metadata
        outfh.write(HTML_FOOTER)
    with open(outdir / "style.css", "wt") as outfh:
        LOGGER.info("Génération de %s", outdir / "style.css")
        outfh.write(STYLE_CSS)
    return metadata


def main(args) -> None:
    extracteur = Extracteur()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    pdfdata = {}
    if args.metadata:
        with open(args.metadata, "rt") as infh:
            pdfdata = json.load(infh)
    metadata["pdfs"] = pdfdata
    docs = []
    for path in args.docs:
        pdf_path = path.with_suffix(".pdf")
        if pdfdata and pdf_path.name not in pdfdata:
            LOGGER.warning("Non-traitement de %s car absent des metadonnées", path)
            continue
        conv = None
        if path.suffix == ".csv":
            LOGGER.info("Lecture de %s", path)
            iob = list(read_csv(path))
        elif path.suffix == ".pdf":
            csvpath = path.with_suffix(".csv")
            if not args.no_csv and csvpath.exists():
                LOGGER.info("Lecture de %s", csvpath)
                iob = list(read_csv(csvpath))
            else:
                LOGGER.info("Conversion, segmentation et classification de %s", path)
                conv = Converteur(path)
                feats = conv.extract_words()
                if args.segment_model is None:
                    if conv.tree is None:
                        LOGGER.warning("Structure logique absente: %s", path)
                        crf = Segmenteur(DEFAULT_MODEL_NOSTRUCT)
                    else:
                        crf = Segmenteur(DEFAULT_SEGMENT_MODEL)
                else:
                    crf = Segmenteur(args.segment_model)
                iob = list(extracteur(crf(feats)))

        if conv is None and pdf_path.exists():
            conv = Converteur(pdf_path)
        doc = extract_html(args, path, iob, conv)
        if pdfdata:
            doc.pdfurl = pdfdata.get(pdf_path.name, {}).get("url", None)
        docs.append(doc)
        if "zonage" in doc.titre.lower() and "zonage" not in metadata:
            metadata["zonage"] = extract_zonage(doc)
    # Create the full tree first to gather information for linking
    metadata["doc"] = make_doc_tree(docs, args.outdir)
    # Detect and resolve links in the text
    extract_links(docs, metadata)
    # Now finally output the text itself
    for doc in docs:
        output_html(args, path, doc)
    with open(args.outdir / "index.json", "wt") as outfh:
        LOGGER.info("Génération de %s", args.outdir / "index.json")
        json.dump(metadata, outfh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    # Done by top-level alexi if not running this as script
    parser.add_argument(
        "-v", "--verbose", help="Notification plus verbose", action="store_true"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    main(args)
