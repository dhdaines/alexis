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
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

from alexi.analyse import Analyseur, Bloc, Document, Element, extract_zonage
from alexi.convert import Converteur
from alexi.format import HtmlFormatter
from alexi.label import DEFAULT_MODEL as DEFAULT_LABEL_MODEL
from alexi.label import Identificateur
from alexi.link import Resolver
from alexi.segment import DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL
from alexi.segment import DEFAULT_MODEL_NOSTRUCT, Segmenteur
from alexi.types import T_obj

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
#header a:link {
    color: #fff;
}
#header a:visited {
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
    prev_level = 0
    for parts, el in doc.structure.traverse():
        if el.type in ("Article", "Annexe"):
            eldir = Path(doc.fileid, el.type, el.numero)
        else:
            eldir = Path(doc.fileid, *parts, el.type, el.numero)
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
        level = len(parts) / 2
        while level < prev_level:
            outfh.write("</ul></details></li>\n")
            prev_level -= 1
        pdflink = ""
        if doc.pdfurl is not None:
            pdflink = (
                f' (<a target="_blank" href="{doc.pdfurl}#page={el.page}">PDF</a>)'
            )
        if el.sub:
            outfh.write(
                f'<li class="{el.type} node"><details><summary>{eltitre}</summary><ul>\n'
            )
            link = f'<a target="_blank" href="{eldir}/index.html">Texte intégral</a>'
            outfh.write(f'<li class="text">{link}{pdflink}</li>\n')
        else:
            link = f'<a target="_blank" href="{eldir}/index.html">{eltitre}</a>'
            outfh.write(f'<li class="{el.type} leaf">{link}{pdflink}</li>\n')
        prev_level = level
    while prev_level > 0:
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
            # Make fragment links to this ID expand the document (as
            # we usually do not want to link to the full text)
            outfh.write(
                f'<summary id="{doc.fileid}">{doc.numero}: {doc.titre}</summary>\n'
            )
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


class Extracteur:
    def __init__(
        self,
        outdir: Path,
        metadata: Path,
        segment_model: Optional[Path] = None,
        no_csv=False,
        no_images=False,
    ):
        self.outdir = outdir
        self.crf_s = Identificateur()
        if segment_model is not None:
            self.crf = Segmenteur(segment_model)
            self.crf_n = None
        else:
            self.crf = Segmenteur(DEFAULT_SEGMENT_MODEL)
            self.crf_n = Segmenteur(DEFAULT_MODEL_NOSTRUCT)
        if metadata:
            with open(metadata, "rt") as infh:
                self.pdfdata = json.load(infh)
        else:
            self.pdfdata = {}
        self.metadata = {"pdfs": self.pdfdata}
        self.no_csv = no_csv
        self.no_images = no_images
        outdir.mkdir(parents=True, exist_ok=True)

    def __call__(self, path: Path) -> Optional[Document]:
        pdf_path = path.with_suffix(".pdf")
        if self.pdfdata and pdf_path.name not in self.pdfdata:
            LOGGER.warning("Non-traitement de %s car absent des metadonnées", path)
            return None
        conv = None
        if path.suffix == ".csv":
            LOGGER.info("Lecture de %s", path)
            iob = list(read_csv(path))
        elif path.suffix == ".pdf":
            csvpath = path.with_suffix(".csv")
            if not self.no_csv and csvpath.exists():
                LOGGER.info("Lecture de %s", csvpath)
                iob = list(read_csv(csvpath))
            else:
                LOGGER.info("Conversion, segmentation et classification de %s", path)
                conv = Converteur(path)
                feats = conv.extract_words()
                crf = self.crf
                if conv.tree is None:
                    LOGGER.warning("Structure logique absente: %s", path)
                    if self.crf_n is not None:
                        crf = self.crf_n
                iob = list(self.crf_s(crf(feats)))
        if conv is None and pdf_path.exists():
            conv = Converteur(pdf_path)
        doc = self.analyse(iob, conv, path.stem)
        if self.pdfdata:
            doc.pdfurl = self.pdfdata.get(pdf_path.name, {}).get("url", None)
        if "zonage" in doc.titre.lower() and "zonage" not in self.metadata:
            self.metadata["zonage"] = extract_zonage(doc)
        return doc

    def analyse(self, iob: Iterable[T_obj], conv: Converteur, fileid: str):
        docdir = self.outdir / fileid
        imgdir = self.outdir / fileid / "img"
        LOGGER.info("Génération de pages HTML sous %s", docdir)
        docdir.mkdir(parents=True, exist_ok=True)
        analyseur = Analyseur(fileid, iob)
        if conv and not self.no_images:
            LOGGER.info("Extraction d'images sous %s", imgdir)
            imgdir.mkdir(parents=True, exist_ok=True)
            images = conv.extract_images()
            analyseur.add_images(images)
            save_images_from_pdf(analyseur.blocs, conv, imgdir)
        LOGGER.info("Analyse de la structure de %s", fileid)
        return analyseur()

    def output_json(self):
        """Sauvegarder les metadonnées"""
        with open(self.outdir / "index.json", "wt") as outfh:
            LOGGER.info("Génération de %s", self.outdir / "index.json")
            json.dump(self.metadata, outfh, indent=2, ensure_ascii=False)

    def output_doctree(self, docs: list[Document]):
        """Générer la page HTML principale et créer l'index de documents."""
        self.metadata["docs"] = make_doc_tree(docs, self.outdir)
        self.resolver = Resolver(self.metadata)

    def output_section_index(self, doc: Document, path: Path, elements: list[Element]):
        """Générer l'index de textes à un palier (Article, Annexe, etc)"""
        doc_titre = doc.titre if doc.titre != "Document" else doc.fileid
        title = f"{doc_titre}: {path.name}s"
        docdir = self.outdir / doc.fileid / path
        style = os.path.relpath(self.outdir / "style.css", docdir)
        HTML_HEADER = (
            HTML_GLOBAL_HEADER
            + f"""    <link rel="stylesheet" href="{style}">
    <title>{title}</title>
  </head>
  <body>
    <div class="container">
    <h1 id="header">{title}</h1>
    <ul id="body">\n"""
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

    def output_sub_index(self, doc: Document, el: Element, path: Path):
        subtypes = list(el.sub)
        gt = operator.attrgetter("type")
        subtypes.sort(key=gt)
        for subtype, elements in itertools.groupby(subtypes, gt):
            if subtype not in ("Article", "Annexe"):
                self.output_section_index(doc, path / subtype, elements)

    def output_section(self, doc: Document, path: Path, elements: list[Element]):
        if not elements:
            return
        self.output_section_index(doc, path, elements)
        for el in elements:
            self.output_element(doc, path / el.numero, el)

    def output_element(self, doc: Document, path: Path, el: Element):
        """Générer le HTML for un seul élément."""
        docdir = self.outdir / doc.fileid
        imgdir = docdir / "img"
        outdir = docdir / path
        # Can't use Path.relative_to until 3.12 :(
        rel_imgdir = os.path.relpath(imgdir, outdir)
        rel_style = os.path.relpath(self.outdir / "style.css", outdir)
        doc_titre = el.titre
        if doc.titre != "Document":
            doc_titre = doc.titre
            if doc.numero:
                doc_titre = f'{doc.numero} <span class="nomobile">{doc.titre}</span>'
        pdflink = ""
        if doc.pdfurl is not None:
            pdflink = f' (<a target="_blank" href="{doc.pdfurl}">PDF</a>)'
        HTML_HEADER = (
            HTML_GLOBAL_HEADER
            + f"""    <link rel="stylesheet" href="{rel_style}">
    <title>{el.titre}</title>
  </head>
  <body>
    <div class="container">
    <h1 id="header">{doc_titre}{pdflink}</h1>
    <div id="body">
    """
        )
        HTML_FOOTER = """</div></div></body>
</html>
"""
        outdir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Génération %s %s -> %s/index.html (PDF: %s#%d)",
            el.type,
            el.numero,
            outdir,
            doc.pdfurl,
            el.page,
        )
        formatter = HtmlFormatter(
            doc=doc, imgdir=rel_imgdir, resolver=self.resolver, path=path
        )
        with open(outdir / "index.html", "wt") as outfh:
            outfh.write(HTML_HEADER)
            outfh.write(formatter(element=el, fragment=True))
            outfh.write(HTML_FOOTER)

    def output_html(self, doc: Document):
        # Do articles/annexes at top level (FIXME: parameterize this)
        for palier in ("Article", "Annexe"):
            self.output_section(doc, Path(palier), doc.paliers.get(palier, []))

        # Do the top directory with full text rather than an index
        path = Path(".")
        if doc.structure.sub:
            self.output_sub_index(doc, doc.structure, path)
        self.output_element(doc, path, doc.structure)

        for parts, el in doc.structure.traverse():
            if el.type in ("Article", "Annexe"):
                continue
            LOGGER.info(
                "Path: %s Structure: %s %s [%s]",
                parts,
                el.type,
                el.numero,
                ",".join("%s %s" % (sel.type, sel.numero) for sel in el.sub),
            )
            self.output_element(doc, Path(*parts, el.type, el.numero), el)
        return doc


def main(args) -> None:
    extracteur = Extracteur(
        args.outdir, args.metadata, args.segment_model, args.no_csv, args.no_images
    )
    docs = []
    for path in args.docs:
        doc = extracteur(path)
        if doc is not None:
            docs.append(doc)
    extracteur.output_doctree(docs)
    for doc in docs:
        extracteur.output_html(doc)
    extracteur.output_json()


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
