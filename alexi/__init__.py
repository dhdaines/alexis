"""
ALexi, EXtracteur d'Information

Ce module est le point d'entrée principale pour le logiciel ALEXI.
"""

import argparse
import csv
import dataclasses
import itertools
import json
import logging
import operator
import sys
from pathlib import Path

from . import annotate, download, extract
from .analyse import Analyseur, Bloc, merge_overlaps
from .convert import Converteur, write_csv
from .format import format_html
from .index import index
from .label import DEFAULT_MODEL as DEFAULT_LABEL_MODEL
from .label import Identificateur
from .search import search
from .segment import DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL
from .segment import Segmenteur

LOGGER = logging.getLogger("alexi")
VERSION = "0.4.0"


def convert_main(args: argparse.Namespace):
    """Convertir les PDF en CSV"""
    if args.pages:
        pages = [max(1, int(x)) for x in args.pages.split(",")]
    else:
        pages = None
    conv = Converteur(args.pdf)
    if args.images is not None:
        args.images.mkdir(parents=True, exist_ok=True)
        images: list[dict] = []
        for _, group in itertools.groupby(
            conv.extract_images(pages), operator.attrgetter("page_number")
        ):
            merged = merge_overlaps(group)
            for bloc in merged:
                images.append(dataclasses.asdict(bloc))
                img = (
                    conv.pdf.pages[bloc.page_number - 1]
                    .crop(bloc.bbox)
                    .to_image(resolution=150, antialias=True)
                )
                LOGGER.info("Extraction de %s", args.images / bloc.img)
                img.save(args.images / bloc.img)
        with open(args.images / "images.json", "wt") as outfh:
            json.dump(images, outfh, indent=2)
    write_csv(conv.extract_words(pages), sys.stdout)


def segment_main(args: argparse.Namespace):
    """Segmenter un CSV"""
    crf: Segmenteur
    crf = Segmenteur(args.model)
    reader = csv.DictReader(args.csv)
    write_csv(crf(reader), sys.stdout)


def label_main(args: argparse.Namespace):
    """Étiquetter un CSV"""
    crf = Identificateur(args.model)
    reader = csv.DictReader(args.csv)
    write_csv(crf(reader), sys.stdout)


def html_main(args: argparse.Namespace):
    """Convertir un CSV segmenté et étiquetté en HTML"""
    reader = csv.DictReader(args.csv)
    analyseur = Analyseur(args.csv.name, reader)
    if args.images is not None:
        with open(args.images / "images.json", "rt") as infh:
            images = (Bloc(**image_dict) for image_dict in json.load(infh))
            analyseur.add_images(images, merge=False)
        doc = analyseur()
        print(format_html(doc, imgdir=args.images))
    else:
        doc = analyseur()
        print(format_html(doc))


def json_main(args: argparse.Namespace):
    """Convertir un CSV segmenté et étiquetté en JSON"""
    iob = csv.DictReader(args.csv)
    analyseur = Analyseur(args.csv.name, iob)
    if args.images:
        with open(args.images / "images.json", "rt") as infh:
            images = [Bloc(**image_dict) for image_dict in json.load(infh)]
            doc = analyseur(images)
    else:
        doc = analyseur()
    print(json.dumps(dataclasses.asdict(doc), indent=2, ensure_ascii=False))


def index_main(args: argparse.Namespace):
    """Construire un index sur des fichiers JSON"""
    index(args.indir, args.outdir)


def search_main(args: argparse.Namespace):
    """Lancer une recherche sur l'index"""
    search(args.indexdir, args.query, args.nresults)


def make_argparse() -> argparse.ArgumentParser:
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    subp = parser.add_subparsers(required=True)
    download_command = subp.add_parser(
        "download", help="Télécharger les documents plus récents du site web"
    )
    download.add_arguments(download_command)
    download_command.set_defaults(func=download.main)

    convert = subp.add_parser(
        "convert", help="Convertir le texte et les objets des fichiers PDF en CSV"
    )
    convert.add_argument(
        "pdf", help="Fichier PDF à traiter", type=argparse.FileType("rb")
    )
    convert.add_argument(
        "--pages", help="Liste de numéros de page à extraire, séparés par virgule"
    )
    convert.add_argument(
        "--images", help="Répertoire pour écrire des images des tableaux", type=Path
    )
    convert.set_defaults(func=convert_main)

    segment = subp.add_parser(
        "segment", help="Segmenter et étiquetter les segments d'un CSV"
    )
    segment.add_argument(
        "--model", help="Modele CRF", type=Path, default=DEFAULT_SEGMENT_MODEL
    )
    segment.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    segment.set_defaults(func=segment_main)

    label = subp.add_parser(
        "label", help="Étiquetter (extraire des informations) un CSV segmenté"
    )
    label.add_argument(
        "--model", help="Modele CRF", type=Path, default=DEFAULT_LABEL_MODEL
    )
    label.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    label.set_defaults(func=label_main)

    html = subp.add_parser(
        "html",
        help="Extraire la structure en format HTML en partant du CSV étiquetté",
    )
    html.add_argument("csv", help="Fichier CSV à traiter", type=argparse.FileType("rt"))
    html.add_argument(
        "--images", help="Répertoire avec des images des tableaux", type=Path
    )
    html.set_defaults(func=html_main)

    jsonf = subp.add_parser(
        "json",
        help="Extraire la structure en format JSON en partant du CSV étiquetté",
    )
    jsonf.add_argument(
        "csv", help="Fichier CSV à traiter", type=argparse.FileType("rt")
    )
    jsonf.add_argument(
        "--images", help="Répertoire contenant les images des tableaux", type=Path
    )
    jsonf.set_defaults(func=json_main)

    extract_command = subp.add_parser(
        "extract",
        help="Extraire la structure complète de fichiers PDF",
    )
    extract.add_arguments(extract_command)
    extract_command.set_defaults(func=extract.main)

    index = subp.add_parser(
        "index", help="Générer un index Whoosh sur les documents extraits"
    )
    index.add_argument(
        "-o",
        "--outdir",
        help="Repertoire destination pour l'index",
        type=Path,
        default="export/_idx",
    )
    index.add_argument("indir", help="Repertoire avec les fichiers extraits", type=Path)
    index.set_defaults(func=index_main)

    search = subp.add_parser("search", help="Effectuer une recherche sur l'index")
    search.add_argument(
        "-i",
        "--indexdir",
        help="Repertoire source pour l'index",
        type=Path,
        default="export/_idx",
    )
    search.add_argument(
        "-n", "--nresults", help="Nombre de résultats affichés", type=int, default=10
    )
    search.add_argument("query", help="Requête", nargs="+")
    search.set_defaults(func=search_main)

    annotate_command = subp.add_parser(
        "annotate", help="Annoter un PDF pour corriger erreurs"
    )
    annotate.add_arguments(annotate_command)
    annotate_command.set_defaults(func=annotate.main)

    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(filename)s:%(lineno)d (%(funcName)s):%(levelname)s:%(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
