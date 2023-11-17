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
from typing import Any, Iterable, Iterator

from alexi.types import Bloc, T_bbox
from alexi.analyse import Analyseur, group_iob
from alexi.convert import Converteur, bbox_contains, bbox_overlaps
from alexi.segment import Segmenteur
from alexi.format import format_html, format_text, format_dict
from alexi.label import Extracteur

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


def extract_images(blocs: list[Bloc], conv: Converteur, docdir: Path) -> Iterator[Bloc]:
    images = {}
    # Find images in tagged text
    for bloc in blocs:
        if bloc.type in ("Tableau", "Figure"):
            images.setdefault(bloc.page_number, []).append(bloc)
    # Replace tag blocs with structure blocs if possible
    struct_blocs = list(conv.extract_images())
    replace_blocs = expand_text_blocs(images, struct_blocs)
    # Replace "expanded" blocs
    for idx, bloc in enumerate(blocs):
        if bloc in replace_blocs:
            blocs[idx] = replace_blocs[bloc]
    # Find ones not linked to any text and not contained in existing blocs
    insert_blocs = {}
    for bloc in struct_blocs:
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
        blocs = extract_images(blocs, conv, imgdir)
        doc = analyseur(iob, blocs)
    else:
        LOGGER.info("Analyse de la structure de %s", path)
        doc = analyseur(iob)
    with open(docdir / f"{path.stem}.json", "wt") as outfh:
        LOGGER.info("Génération de %s/%s.json", docdir, path.stem)
        docdict = format_dict(doc, imgdir=path.stem)
        pdf_path = path.with_suffix(".pdf")
        docdict["fichier"] = pdf_path.name
        json.dump(docdict, outfh, indent=2, ensure_ascii=False)


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
        blocs = extract_images(blocs, conv, imgdir)
        doc = analyseur(iob, blocs)
    else:
        LOGGER.info("Analyse de la structure de %s", path)
        doc = analyseur(iob)

    def extract_element(el, outdir, fragment=True):
        """Extract the various constituents, referencing images in the
        generated image directory."""
        outdir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("%s %s", outdir, el.titre)
        # Can't use Path.relative_to until 3.12 :(
        rel_imgdir = os.path.relpath(imgdir, outdir)
        with open(outdir / "index.html", "wt") as outfh:
            outfh.write(
                format_html(doc, element=el, imgdir=rel_imgdir, fragment=fragment)
            )
        with open(outdir / "index.md", "wt") as outfh:
            outfh.write(format_text(doc, element=el))
        with open(outdir / "index.json", "wt") as outfh:
            json.dump(dataclasses.asdict(el), outfh)

    # Do articles/annexes at top level
    seen_paliers = set()
    for palier in ("Article", "Annexe"):
        if palier not in doc.paliers:
            continue
        seen_paliers.add(palier)
        # These go in the top level
        for idx, el in enumerate(doc.paliers[palier]):
            extract_element(el, docdir / palier / el.numero)
    # Now do the rest of the Document hierarchy if it exists
    top = Path(docdir)
    d = deque((el, idx, top) for idx, el in enumerate(doc.structure.sub))
    while d:
        el, idx, parent = d.popleft()
        if el.type in seen_paliers:
            continue
        extract_element(el, parent / el.type / el.numero)
        d.extendleft(
            (subel, idx, parent / el.type / el.numero)
            for idx, subel in reversed(list(enumerate(el.sub)))
        )
    # And do a full extraction (which might crash your browser)
    extract_element(doc.structure, docdir, fragment=False)


def main(args):
    crf = None
    extracteur = Extracteur()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in args.docs:
        conv = None
        if path.suffix == ".csv":
            LOGGER.info("Lecture de %s", path)
            iob = list(read_csv(path))
        elif path.suffix == ".pdf":
            LOGGER.info("Conversion, segmentation et classification de %s", path)
            conv = Converteur(path)
            feats = conv.extract_words()
            if crf is None:
                crf = Segmenteur()
            iob = list(extracteur(crf(feats)))

        pdf_path = path.with_suffix(".pdf")
        if conv is None and pdf_path.exists():
            conv = Converteur(pdf_path)
        if args.serafim:
            extract_serafim(args, path, iob, conv)
        else:
            extract_html(args, path, iob, conv)


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
