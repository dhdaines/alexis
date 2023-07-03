"""
Lancer des recherches dans l'index de données.
"""

import argparse
from pathlib import Path
from typing import List

from whoosh.index import open_dir  # type: ignore
from whoosh.qparser import OrGroup, MultifieldParser  # type: ignore

from alexi.types import Reglement


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--indexdir",
        help="Repertoire source pour l'index",
        type=Path,
        default="indexdir",
    )
    parser.add_argument("query", help="Requête", nargs="+")
    return parser


def search(indexdir: Path, terms: List[str]):
    ix = open_dir(indexdir)
    parser = MultifieldParser(["titre", "contenu"], ix.schema, group=OrGroup.factory(0.9))
    query = parser.parse(" ".join(terms))
    with ix.searcher() as searcher:
        results = searcher.search(query)
        for r in results:
            print(r.score, r["titre"])
            print()
        

def main():
    parser = make_argparse()
    args = parser.parse_args()
    search(args.indexdir, args.query)

if __name__ == "__main__":
    main()
