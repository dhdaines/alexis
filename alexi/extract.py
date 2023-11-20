"""
Convertir les règlements en HTML, texte, et/ou JSON structuré.
"""

import argparse
import csv
import dataclasses
import itertools
import json
import logging
import operator
import os
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from alexi.types import Bloc, T_bbox
from alexi.analyse import Analyseur, group_iob, Element, Document
from alexi.convert import Converteur, bbox_contains, bbox_overlaps
from alexi.segment import Segmenteur, DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL
from alexi.format import format_html, format_text, format_dict
from alexi.label import Extracteur, DEFAULT_MODEL as DEFAULT_LABEL_MODEL

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
    parser.add_argument(
        "--segment-model", help="Modele CRF", type=Path, default=DEFAULT_SEGMENT_MODEL
    )
    parser.add_argument(
        "--label-model", help="Modele CRF", type=Path, default=DEFAULT_LABEL_MODEL
    )
    parser.add_argument(
        "-s",
        "--serafim",
        help="Générer le format JSON attendu par SÈRAFIM",
        action="store_true",
    )
    parser.add_argument(
        "docs", help="Documents en PDF ou CSV pré-annoté", type=Path, nargs="+"
    )
    return parser


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "rt") as infh:
        return list(csv.DictReader(infh))


def expand_text_blocs(images: dict[int, list[Bloc]], structure_blocs: Iterable[Bloc]):
    bloclist = list(structure_blocs)
    replace_blocs = {}
    for page_number, blocs in images.items():
        for text_bloc in blocs:
            for struct_bloc in bloclist:
                if text_bloc.page_number != struct_bloc.page_number:
                    continue
                if bbox_contains(struct_bloc.bbox, text_bloc.bbox):
                    struct_bloc.contenu.extend(text_bloc.contenu)
                    LOGGER.info(
                        "page %d replace %s => %s",
                        page_number,
                        text_bloc.bbox,
                        struct_bloc.bbox,
                    )
                    replace_blocs[text_bloc] = struct_bloc
        images[page_number] = [replace_blocs.get(b, b) for b in blocs]
    return replace_blocs


def bbox_above(bbox: T_bbox, other: T_bbox) -> bool:
    """Déterminer si une BBox se trouve en haut d'une autre."""
    _, _, _, bottom = bbox
    _, top, _, _ = other
    return bottom <= top


def bbox_below(bbox: T_bbox, other: T_bbox) -> bool:
    """Déterminer si une BBox se trouve en bas d'une autre."""
    _, top, _, _ = bbox
    _, _, _, bottom = other
    return top >= bottom


def bbox_between(bbox: T_bbox, a: T_bbox, b: T_bbox) -> bool:
    """Déterminer si une BBox se trouve entre deux autres."""
    _, top, _, bottom = bbox
    return top >= a[1] and bottom <= b[3]


def insert_outside_blocs(
    blocs: list[Bloc], insert_blocs: dict[int, list[Bloc]]
) -> Iterator[Bloc]:
    for page, group in itertools.groupby(blocs, operator.attrgetter("page_number")):
        if page not in insert_blocs:
            yield from group
            continue
        page_blocs = list(group)
        page_insert_blocs = insert_blocs[page]
        if len(page_blocs) == 0:
            yield from page_insert_blocs
            continue
        top_bloc = page_blocs[0]
        bottom_bloc = page_blocs[-1]
        inserted_blocs = set()
        for bloc in page_insert_blocs:
            if bloc in inserted_blocs:
                continue
            if bbox_above(bloc.bbox, top_bloc.bbox):
                LOGGER.info(
                    "inserted non-text bloc %s at top of page %d",
                    bloc.bbox,
                    bloc.page_number,
                )
                inserted_blocs.add(bloc)
                yield bloc
        for bloc_a, bloc_b in itertools.pairwise(page_blocs):
            yield bloc_a
            for bloc in page_insert_blocs:
                if bloc in inserted_blocs:
                    continue
                if bbox_between(bloc.bbox, bloc_a.bbox, bloc_b.bbox):
                    LOGGER.info(
                        "inserted non-text bloc %s inside page %d",
                        bloc.bbox,
                        bloc.page_number,
                    )
                    inserted_blocs.add(bloc)
                    yield bloc
                elif bbox_overlaps(bloc.bbox, bloc_a.bbox) and bbox_above(
                    bloc.bbox, bloc_b.bbox
                ):
                    LOGGER.info(
                        "inserted non-text bloc %s (overlaps %s, above %s) page %d",
                        bloc.bbox,
                        bloc_a.bbox,
                        bloc_b.bbox,
                        bloc.page_number,
                    )
                    inserted_blocs.add(bloc)
                    yield bloc
                elif bbox_overlaps(bloc.bbox, bloc_b.bbox) and bbox_below(
                    bloc.bbox, bloc_a.bbox
                ):
                    LOGGER.info(
                        "inserted non-text bloc %s (overlaps %s, below %s) page %d",
                        bloc.bbox,
                        bloc_b.bbox,
                        bloc_a.bbox,
                        bloc.page_number,
                    )
                    inserted_blocs.add(bloc)
                    yield bloc
        yield bloc_b
        for bloc in page_insert_blocs:
            if bloc in inserted_blocs:
                continue
            if bbox_below(bloc.bbox, bottom_bloc.bbox):
                LOGGER.info(
                    "inserted non-text bloc %s at bottom page %d",
                    bloc.bbox,
                    bloc.page_number,
                )
                yield bloc


def insert_images_from_pdf(
    blocs: list[Bloc], conv: Converteur, docdir: Path
) -> Iterator[Bloc]:
    """Convertir des éléments du PDF difficiles à réaliser en texte en
    images et les insérer dans la structure du document (liste de blocs).
    """
    images: dict[int, list[Bloc]] = {}
    # Find images in tagged text
    for bloc in blocs:
        if bloc.type in ("Tableau", "Figure"):
            assert isinstance(bloc.page_number, int)
            images.setdefault(bloc.page_number, []).append(bloc)
    # Replace tag blocs with structure blocs if possible
    struct_blocs = list(conv.extract_images())
    replace_blocs = expand_text_blocs(images, struct_blocs)
    # Replace "expanded" blocs
    for idx, bloc in enumerate(blocs):
        if bloc in replace_blocs:
            blocs[idx] = replace_blocs[bloc]
    # Find ones not linked to any text and not contained in existing blocs
    insert_blocs: dict[int, list[Bloc]] = {}
    for bloc in struct_blocs:
        assert isinstance(bloc.page_number, int)
        if bloc.contenu:
            continue
        is_contained = False
        for outer_bloc in images.setdefault(bloc.page_number, []):
            if bbox_contains(outer_bloc.bbox, bloc.bbox):
                is_contained = True
                break
        for outer_bloc in struct_blocs:
            if outer_bloc is bloc or outer_bloc.page_number != bloc.page_number:
                continue
            if bbox_contains(outer_bloc.bbox, bloc.bbox):
                is_contained = True
                break
        if not is_contained:
            insert_blocs.setdefault(bloc.page_number, []).append(bloc)
            images[bloc.page_number].append(bloc)
    # Render image files
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
    # Insert non-text blocks into the document for reparsing
    return insert_outside_blocs(blocs, insert_blocs)


def extract_serafim(args, path, iob, conv):
    docdir = args.outdir / "data"
    imgdir = args.outdir / "public" / "img" / path.stem
    LOGGER.info("Génération de fichiers SÈRAFIM sous %s", docdir)
    docdir.mkdir(parents=True, exist_ok=True)
    analyseur = Analyseur()
    if not args.no_images:
        LOGGER.info("Extraction d'images sous %s", imgdir)
        imgdir.mkdir(parents=True, exist_ok=True)
    if conv and not args.no_images:
        LOGGER.info("Extraction d'images de %s", path)
        blocs = list(group_iob(iob))
        blocs = insert_images_from_pdf(blocs, conv, imgdir)
        doc = analyseur(path.stem, iob, blocs)
    else:
        LOGGER.info("Analyse de la structure de %s", path)
        doc = analyseur(path.stem, iob)
    with open(docdir / f"{path.stem}.json", "wt") as outfh:
        LOGGER.info("Génération de %s/%s.json", docdir, path.stem)
        docdict = format_dict(doc, imgdir=path.stem)
        pdf_path = path.with_suffix(".pdf")
        docdict["fichier"] = pdf_path.name
        json.dump(docdict, outfh, indent=2, ensure_ascii=False)


HTML_GLOBAL_HEADER = """<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/purecss@3.0.0/build/pure-min.css" integrity="sha384-X38yfunGUhNzHpBaEBsWLO+A0HDYOQi8ufWDkZ0k9e0eXz/tH3II7uKZ9msv++Ls" crossorigin="anonymous">
"""


def extract_element(
    doc: Document, el: Element, outdir: Path, imgdir: Path, fragment=True
):
    """Extract the various constituents, referencing images in the
    generated image directory."""
    # Can't use Path.relative_to until 3.12 :(
    rel_imgdir = os.path.relpath(imgdir, outdir)
    rel_style = os.path.relpath(imgdir.parent.parent / "style.css", outdir)
    HTML_HEADER = (
        HTML_GLOBAL_HEADER
        + f"""    <link rel="stylesheet" href="{rel_style}">
    <title>{el.titre}</title>
  </head>
  <body>
    <div id="body">
"""
    )
    HTML_FOOTER = """</div></body>
</html>
"""
    outdir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("%s %s", outdir, el.titre)
    with open(outdir / "index.html", "wt") as outfh:
        outfh.write(HTML_HEADER)
        outfh.write(format_html(doc, element=el, imgdir=rel_imgdir, fragment=fragment))
        outfh.write(HTML_FOOTER)
    with open(outdir / "index.md", "wt") as outfh:
        outfh.write(format_text(doc, element=el))
    with open(outdir / "index.json", "wt") as outfh:
        json.dump(dataclasses.asdict(el), outfh)


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


def extract_html(args, path, iob, conv):
    docdir = args.outdir / path.stem
    imgdir = args.outdir / path.stem / "img"
    LOGGER.info("Génération de pages HTML sous %s", docdir)
    docdir.mkdir(parents=True, exist_ok=True)
    analyseur = Analyseur()
    if conv and not args.no_images:
        LOGGER.info("Extraction d'images sous %s", imgdir)
        imgdir.mkdir(parents=True, exist_ok=True)
        blocs = list(group_iob(iob))
        blocs = insert_images_from_pdf(blocs, conv, imgdir)
        doc = analyseur(path.stem, iob, blocs)
    else:
        LOGGER.info("Analyse de la structure de %s", path)
        doc = analyseur(path.stem, iob)

    if doc.numero and doc.numero != path.stem:
        LOGGER.info("Lien %s => %s", doc.numero, path.stem)
        Path(args.outdir / doc.numero).symlink_to(path.stem)
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
    extract_element(doc, doc.structure, docdir, imgdir, fragment=False)
    return doc


def make_doc_subtree(doc: Document, outfh: TextIO):
    outfh.write("<ul>\n")
    outfh.write(
        f'<li class="text"><a target="_blank" href="{doc.fileid}/index.html">Texte intégral</a></li>\n'
    )
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
            outfh.write("</ul></li>\n")
            prev_level -= 1
        if el.sub:
            outfh.write(f'<li class="node"><details><summary>{eltitre}</summary><ul>\n')
            link = f'<a target="_blank" href="{eldir}/index.html">Texte intégral</a>'
            outfh.write(f'<li class="text">{link}</li>\n')
        else:
            link = f'<a target="_blank" href="{eldir}/index.html">{eltitre}</a>'
            outfh.write(f'<li class="leaf">{link}</li>\n')
        d.extendleft((subel, eldir, level + 1) for subel in reversed(el.sub))
        prev_level = level
    while prev_level > 1:
        outfh.write("</ul></li>\n")
        prev_level -= 1
    outfh.write("</ul>\n")


def make_doc_tree(docs: list[Document], outdir: Path):
    HTML_HEADER = (
        HTML_GLOBAL_HEADER
        + """    <title>ALEXI, EXtracteur d'Information</title>
    <link rel="stylesheet" href="./style.css">
  </head>
  <body>
    <div class="container">
    <h1 id="header">ALEXI, EXtracteur d'Information</h1>
    <ul id="body">
"""
    )
    HTML_FOOTER = """</ul>
    </div>
  </body>
</html>
"""
    docs.sort(key=operator.attrgetter("numero"))
    with open(outdir / "index.html", "wt") as outfh:
        LOGGER.info("Génération de %s", outdir / "index.html")
        outfh.write(HTML_HEADER)
        for doc in docs:
            outfh.write('<li class="node"><details>\n')
            outfh.write(f"<summary>{doc.numero}: {doc.titre}</summary>\n")
            make_doc_subtree(doc, outfh)
            outfh.write("</li>\n")
        outfh.write(HTML_FOOTER)
    with open(outdir / "style.css", "wt") as outfh:
        outfh.write(
            """html, body {
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
#body {
    overflow-y: scroll;
    padding: 2px;
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
        )


def main(args):
    crf = None
    extracteur = Extracteur()
    args.outdir.mkdir(parents=True, exist_ok=True)
    docs = []
    for path in args.docs:
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
                if crf is None:
                    crf = Segmenteur(args.segment_model)
                iob = list(extracteur(crf(feats)))

        pdf_path = path.with_suffix(".pdf")
        if conv is None and pdf_path.exists():
            conv = Converteur(pdf_path)
        if args.serafim:
            extract_serafim(args, path, iob, conv)
        else:
            docs.append(extract_html(args, path, iob, conv))
    if not args.serafim:
        make_doc_tree(docs, args.outdir)


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
