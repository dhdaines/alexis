"""
ALexi, EXtracteur d'Information

Ce module est le point d'entrée principale pour le logiciel ALEXI.
"""

import argparse
import logging
import re
import subprocess
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup

from .convert import write_csv
from .extract import Extracteur
from .index import index
from .search import search

LOGGER = logging.getLogger("alexi")


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


def convert_main(args):
    """Convertir les PDF en CSV"""
    if args.verbose:
        global tqdm
        tqdm = lambda x: x  # noqa: E731
    with pdfplumber.open(args.infile) as pdf:
        LOGGER.info("processing %s", args.infile)
        write_csv(pdf, args.outfile)


def segment_main(args):
    """Extraire les unités de texte des CSV"""
    pass


def index_main(args):
    index(args.outdir, args.jsons)


def extract_main(args):
    """Extraire la structure de documents à partir de CSV segmentés"""
    conv = Extracteur(fichier=Path(args.csv.name).with_suffix(".pdf"))
    doc = conv(args.csv)
    print(doc.model_dump_json(indent=2, exclude_defaults=True))


def search_main(args):
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
    convert.add_argument("infile", help="Fichier PDF à traiter", type=Path)
    convert.add_argument("outfile", help="Fichier CSV à créer", type=Path)
    convert.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    convert.set_defaults(func=convert_main)

    subp.add_parser(
        "segment", help="Extraire les unités de texte des CSV"
    ).set_defaults(func=segment_main)

    extract = subp.add_parser(
        "extract", help="Extraire la structure des CSV segmentés en format JSON"
    )
    extract.add_argument("csv", help="Fichier CSV à traiter", type=Path)
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
