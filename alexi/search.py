"""
Lancer des recherches dans l'index de données.
"""

import argparse
import json
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from lunr.index import Index  # type: ignore
from lunr.languages import get_nltk_builder  # type: ignore

from alexi.index import unifold

# This is just here to register the necessary pipeline functions
get_nltk_builder(["fr"])


def get_pdf(soup: BeautifulSoup):
    header = soup.select("article.Article > h4")[0]
    link = header.select("a")[0]
    return link.get("href")


def search(indexdir: Path, docdir: Path, terms: List[str], nresults: int) -> None:
    with open(indexdir / "index.json", "rt", encoding="utf-8") as infh:
        index = Index.load(json.load(infh))
    with open(indexdir / "textes.json", "rt", encoding="utf-8") as infh:
        docs = [(url, titre) for (url, titre, *_) in json.load(infh)]
    index.pipeline.add(unifold)
    results = index.search(" ".join(terms))
    for idx, r in enumerate(results):
        if idx == nresults:
            break
        url, titre = docs[int(r["ref"])]
        with open(docdir / url) as infh:
            soup = BeautifulSoup(infh, features="lxml")
            print(f"{get_pdf(soup)} {titre}")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-i",
        "--indexdir",
        help="Repertoire source pour l'index",
        type=Path,
        default="export/_idx",
    )
    parser.add_argument(
        "-d",
        "--docdir",
        help="Repertoire source pour documents",
        type=Path,
        default="export",
    )
    parser.add_argument(
        "-n", "--nresults", help="Nombre de résultats affichés", type=int, default=10
    )
    parser.add_argument("query", help="Requête", nargs="+")
    return parser


def main(args: argparse.Namespace):
    """Lancer une recherche sur l'index"""
    search(args.indexdir, args.docdir, args.query, args.nresults)
