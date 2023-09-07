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
import textwrap
from pathlib import Path
from typing import Any, Iterable, TextIO

from bs4 import BeautifulSoup

from .convert import FIELDNAMES, Converteur
from .crf import CRF
from .index import index
from .json import Formatteur
from .label import Classificateur
from .search import search
from .segment import Segmenteur

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


def iob2xml(words):
    cur_tag = None
    words = []
    for word in words:
        bio, sep, tag = word["tag"].partition("-")
        if bio == "B":
            if cur_tag is not None:
                if words:
                    print("\n".join(textwrap.wrap(" ".join(words))))
                print(f"</{cur_tag}>")
            words = []
            cur_tag = tag
            print(f"<{tag}>")
        if bio != "O":
            words.append(word["text"])


def convert_main(args):
    """Convertir les PDF en CSV"""
    if args.images is not None:
        args.images.mkdir(parents=True, exist_ok=True)
    if args.pages:
        pages = [max(0, int(x) - 1) for x in args.pages.split(",")]
    else:
        pages = None
    conv = Converteur(imgdir=args.images)
    write_csv(conv(args.pdf, pages), sys.stdout)


def crf_main(args):
    """Segmenter les PDF avec CRF"""
    crf = CRF(args.model)
    reader = csv.DictReader(args.csv)
    write_csv(crf(reader), sys.stdout)


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
    conv = Formatteur(fichier=args.name, imgdir=args.images)
    reader = csv.DictReader(args.csv)
    doc = conv(reader)
    print(doc.model_dump_json(indent=2, exclude_defaults=True))


def extract_main(args):
    """Convertir un PDF en JSON"""
    if args.images is not None:
        imgdir = args.images / Path(args.pdf.name).stem
        imgdir.mkdir(parents=True, exist_ok=True)
        converteur = Converteur(imgdir=imgdir)
        formatteur = Formatteur(fichier=Path(args.pdf.name).name, imgdir=imgdir)
    else:
        converteur = Converteur()
        formatteur = Formatteur(fichier=Path(args.pdf.name).name)
    segmenteur = Segmenteur()
    classificateur = Classificateur()

    if args.pages:
        pages = [max(0, int(x) - 1) for x in args.pages.split(",")]
    else:
        pages = None
    doc = converteur(args.pdf, pages)
    doc = segmenteur(doc)
    doc = classificateur(doc)
    doc = formatteur(doc)
    print(doc.model_dump_json(indent=2, exclude_defaults=True))


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
        "--images", help="Répertoire pour écrire des images des tableaux", type=Path
    )
    convert.add_argument(
        "--pages", help="Liste de numéros de page à extraire, séparés par virgule"
    )
    convert.set_defaults(func=convert_main)

    crf = subp.add_parser("crf", help="Segmenter PDF avec un CRF")
    crf.add_argument("model", help="Modele CRF", type=Path)
    crf.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    crf.set_defaults(func=crf_main)

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
    json.add_argument(
        "--images", help="Répertoire où trouver des images de figures", type=Path
    )
    json.add_argument("csv", help="Fichier CSV à traiter", type=argparse.FileType("rt"))
    json.set_defaults(func=json_main)

    extract = subp.add_parser("extract", help="Extractir la structure d'un PDF en JSON")
    extract.add_argument(
        "pdf", help="Fichier PDF à traiter", type=argparse.FileType("rb")
    )
    extract.add_argument(
        "--pages", help="Liste de numéros de page à extraire, séparés par virgule"
    )
    extract.add_argument(
        "--images", help="Répertoire pour écrire des images des tableaux", type=Path
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
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
