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

from .convert import Converteur
from .index import index
from .json import Formatteur
from .label import Classificateur
from .search import search
from .segment import Segmenteur

LOGGER = logging.getLogger("alexi")
FIELDNAMES = [
    "tag",
    "text",
    "page",
    "page_width",
    "page_height",
    "r",
    "g",
    "b",
    "x0",
    "x1",
    "top",
    "bottom",
    "doctop",
]


def download_main(args):
    """Télécharger les fichiers avec wget"""
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
    conv = Converteur()
    write_csv(conv(args.pdf), sys.stdout)


def segment_main(args):
    """Extraire les unités de texte des CSV"""
    seg = Segmenteur()
    reader = csv.DictReader(args.csv)
    write_csv(seg(reader), sys.stdout)


def label_main(args):
    """Étiquetter les unités de texte des CSV"""
    classificateur = Classificateur()
    reader = csv.DictReader(args.csv)
    write_csv(classificateur(reader), sys.stdout)


def json_main(args):
    """Convertir un CSV segmenté en JSON"""
    conv = Formatteur(fichier=args.name)
    reader = csv.DictReader(args.csv)
    doc = conv(reader)
    print(doc.json(indent=2, exclude_defaults=True))


def extract_main(args):
    """Convertir un PDF en JSON"""
    converteur = Converteur()
    segmenteur = Segmenteur()
    classificateur = Classificateur()
    formatteur = Formatteur(fichier=Path(args.pdf.name).name)

    doc = converteur(args.pdf)
    doc = segmenteur(doc)
    doc = classificateur(doc)
    doc = formatteur(doc)
    print(doc.json(indent=2, exclude_defaults=True))


def index_main(args):
    """Construire un index sur des fichiers JSON"""
    index(args.outdir, args.jsons)


def search_main(args):
    """Lancer une recherche sur l'index"""
    search(args.indexdir, args.query)


def make_argparse() -> argparse.ArgumentParser:
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    convert.set_defaults(func=convert_main)

    segment = subp.add_parser("segment", help="Extraire les unités de texte des CSV")
    segment.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    segment.set_defaults(func=segment_main)

    label = subp.add_parser("label", help="Étiquetter les unités de texte dans un CSV")
    label.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    label.set_defaults(func=label_main)

    json = subp.add_parser(
        "json",
        help="Extraire la structure en format JSON en partant du CSV étiquetté",
    )
    json.add_argument(
        "-n", "--name", help="Nom du fichier PDF originel", type=Path, default="INCONNU"
    )
    json.add_argument("csv", help="Fichier CSV à traiter", type=argparse.FileType("rt"))
    json.set_defaults(func=json_main)

    extract = subp.add_parser("extract", help="Extractir la structure d'un PDF en JSON")
    extract.add_argument(
        "pdf", help="Fichier PDF à traiter", type=argparse.FileType("rb")
    )
    extract.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    extract.set_defaults(func=extract_main)

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
    args.func(args)
