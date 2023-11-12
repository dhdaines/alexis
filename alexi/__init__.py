"""
ALexi, EXtracteur d'Information

Ce module est le point d'entrée principale pour le logiciel ALEXI.
"""

import argparse
import csv
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

from bs4 import BeautifulSoup

from .analyse import Analyseur
from .convert import FIELDNAMES, Converteur
from .format import format_html, format_xml
from .index import index
from .search import search
from .segment import DEFAULT_MODEL as DEFAULT_SEGMENT_MODEL, Segmenteur
from .label import DEFAULT_MODEL as DEFAULT_LABEL_MODEL, Extracteur

LOGGER = logging.getLogger("alexi")


def download_main(args):
    """Télécharger les fichiers avec wget"""
    try:
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "--timestamping",
                "--recursive",
                "--level=1",
                "--accept-regex",
                r".*upload/documents/.*\.pdf",
                "https://ville.sainte-adele.qc.ca/publications.php",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as err:
        if err.returncode != 8:
            raise


def select_main(args):
    """Trouver une liste de fichiers dans la page web des documents."""
    with open(args.infile) as infh:
        soup = BeautifulSoup(infh, "lxml")
        for h2 in soup.find_all("h2", string=re.compile(args.section, re.I)):
            ul = h2.find_next("ul")
            for li in ul.find_all("li"):
                path = Path(li.a["href"])
                print(path.relative_to("/"))


def write_csv(
    doc: Iterable[dict[str, Any]], outfh: TextIO, fieldnames: list[str] = FIELDNAMES
):
    writer = csv.DictWriter(outfh, fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(doc)


def convert_main(args):
    """Convertir les PDF en CSV"""
    if args.pages:
        pages = [max(0, int(x) - 1) for x in args.pages.split(",")]
    else:
        pages = None
    conv = Converteur(args.pdf)
    write_csv(conv.extract_words(pages), sys.stdout)


def segment_main(args):
    """Segmenter un CSV"""
    crf = Segmenteur(args.model)
    reader = csv.DictReader(args.csv)
    write_csv(crf(reader), sys.stdout)


def label_main(args):
    """Étiquetter un CSV"""
    crf = Extracteur(args.model)
    reader = csv.DictReader(args.csv)
    write_csv(crf(reader), sys.stdout)


def xml_main(args):
    """Convertir un CSV segmenté et étiquetté en XML"""
    reader = csv.DictReader(args.csv)
    doc = Analyseur()(reader)
    print(format_xml(doc))


def html_main(args):
    """Convertir un CSV segmenté et étiquetté en HTML"""
    reader = csv.DictReader(args.csv)
    doc = Analyseur()(reader)
    print(format_html(doc))


def index_main(args):
    """Construire un index sur des fichiers JSON"""
    index(args.outdir, args.jsons)


def search_main(args):
    """Lancer une recherche sur l'index"""
    search(args.indexdir, args.query)


def make_argparse() -> argparse.ArgumentParser:
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    subp = parser.add_subparsers(required=True)
    subp.add_parser(
        "download", help="Télécharger les documents plus récents du site web"
    ).set_defaults(func=download_main)

    select = subp.add_parser(
        "select", help="Générer la liste de documents pour une ou plusieurs catégories"
    )
    select.add_argument(
        "-i",
        "--infile",
        help="Page HTML avec liste de publications",
        type=Path,
        default="ville.sainte-adele.qc.ca/publications.php",
    )
    select.add_argument(
        "-s",
        "--section",
        help="Expression régulière pour sélectionner la section des documents",
        default=r"règlements",
    )
    select.set_defaults(func=select_main)

    convert = subp.add_parser(
        "convert", help="Convertir le texte et les objets des fichiers PDF en CSV"
    )
    convert.add_argument(
        "pdf", help="Fichier PDF à traiter", type=argparse.FileType("rb")
    )
    convert.add_argument(
        "--pages", help="Liste de numéros de page à extraire, séparés par virgule"
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

    xml = subp.add_parser(
        "xml",
        help="Extraire la structure en format XML en partant du CSV étiquetté",
    )
    xml.add_argument("csv", help="Fichier CSV à traiter", type=argparse.FileType("rt"))
    xml.set_defaults(func=xml_main)

    html = subp.add_parser(
        "html",
        help="Extraire la structure en format HTML en partant du CSV étiquetté",
    )
    html.add_argument("csv", help="Fichier CSV à traiter", type=argparse.FileType("rt"))
    html.set_defaults(func=html_main)

    index = subp.add_parser(
        "index", help="Générer un index Whoosh sur les documents extraits"
    )
    index.add_argument(
        "-o",
        "--outdir",
        help="Repertoire destination pour l'index",
        type=Path,
        default="indexdir",
    )
    index.add_argument("jsons", help="Fichiers JSON", type=Path, nargs="+")
    index.set_defaults(func=index_main)

    search = subp.add_parser("search", help="Effectuer une recherche sur l'index")
    search.add_argument(
        "-i",
        "--indexdir",
        help="Repertoire source pour l'index",
        type=Path,
        default="indexdir",
    )
    search.add_argument("query", help="Requête", nargs="+")
    search.set_defaults(func=search_main)
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
