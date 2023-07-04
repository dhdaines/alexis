"""
ALexi, EXtracteur d'Information

Ce module est le point d'entrée principale pour le logiciel ALEXI.
"""

import argparse
from pathlib import Path

from . import cli


def make_argparse() -> argparse.ArgumentParser:
    """Make the argparse"""
    parser = argparse.ArgumentParser(description=__doc__)
    subp = parser.add_subparsers(required=True)
    subp.add_parser(
        "download", help="Télécharger les documents plus récents du site web"
    ).set_defaults(func=cli.download.main)

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
        default=r"règlements"
    )
    select.set_defaults(func=cli.select.main)

    convert = subp.add_parser(
        "convert", help="Convertir le texte et les objets des fichiers PDF en CSV"
    )
    convert.add_argument("infile", help="Fichier PDF à traiter", type=Path)
    convert.add_argument("outfile", help="Fichier CSV à créer", type=Path)
    convert.add_argument(
        "-v", "--verbose", help="Émettre des messages", action="store_true"
    )
    convert.set_defaults(func=cli.convert.main)

    subp.add_parser(
        "segment", help="Extraire les unités de texte des CSV"
    ).set_defaults(func=cli.segment.main)

    subp.add_parser(
        "extract", help="Extraire la structure des CSV segmentés en format JSON"
    ).set_defaults(func=cli.extract.main)

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
    index.set_defaults(func=cli.index.main)

    search = subp.add_parser("search", help="Effectuer une recherche sur l'index")
    search.add_argument(
        "-i",
        "--indexdir",
        help="Repertoire source pour l'index",
        type=Path,
        default="indexdir",
    )
    search.add_argument("query", help="Requête", nargs="+")
    search.set_defaults(
        func=cli.search.main
    )
    return parser


def main():
    parser = make_argparse()
    args = parser.parse_args()
    args.func(args)
