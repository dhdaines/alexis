#!/usr/bin/env python

"""
Convertir les règlements en HTML structuré.
"""

import argparse
import csv
import itertools
import logging
import operator
from pathlib import Path
from typing import Any, Iterable, Iterator

from alexi.types import Bloc, T_bbox
from alexi.analyse import Analyseur, group_iob
from alexi.convert import Converteur, bbox_contains, bbox_overlaps
from alexi.segment import Segmenteur
from alexi.format import format_html, format_text
from alexi.label import Extracteur

LOGGER = logging.getLogger("convert")


def make_argparse():
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--outdir", help="Repertoire de sortie", type=Path, default="html"
    )
    parser.add_argument(
        "-v", "--verbose", help="Notification plus verbose", action="store_true"
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
        for bloc in image_blocs:
            x0, top, x1, bottom = bloc.bbox
            if x0 == x1 or top == bottom:
                LOGGER.warning("Skipping empty image bbox %s", bloc.bbox)
                continue
            img = (
                conv.pdf.pages[page_number - 1]
                .crop(bloc.bbox)
                .to_image(resolution=150, antialias=True)
            )
            LOGGER.info("Extraction de %s", docdir / bloc.img)
            img.save(docdir / bloc.img)
    # Insert non-text blocks into the document for reparsing
    return insert_outside_blocs(blocs, insert_blocs)


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    crf = None
    extracteur = Extracteur()
    analyseur = Analyseur()
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

        docdir = args.outdir / path.stem
        LOGGER.info("Génération de pages HTML sous %s", docdir)
        docdir.mkdir(exist_ok=True)

        if conv:
            LOGGER.info("Extraction d'images de %s", path)
            blocs = list(group_iob(iob))
            blocs = extract_images(blocs, conv, docdir)
            doc = analyseur(iob, blocs)
        else:
            LOGGER.info("Analyse de la structure de %s", path)
            doc = analyseur(iob)

        for palier, elements in doc.paliers.items():
            for idx, element in enumerate(elements):
                if palier == "Document":
                    element = None
                    title = "index"
                else:
                    title = f"{palier}_{idx+1}"
                with open(docdir / f"{title}.html", "wt") as outfh:
                    LOGGER.info("Génération de %s/%s.html", docdir, title)
                    outfh.write(format_html(doc, element=element))
                with open(docdir / f"{title}.txt", "wt") as outfh:
                    LOGGER.info("Génération de %s/%s.txt", docdir, title)
                    outfh.write(format_text(doc, element=element))


if __name__ == "__main__":
    main()
