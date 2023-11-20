"""
Lancer des recherches dans l'index de données.
"""

from pathlib import Path
from typing import List

from whoosh.index import open_dir  # type: ignore
from whoosh.qparser import MultifieldParser, OrGroup  # type: ignore


def search(indexdir: Path, terms: List[str]) -> None:
    ix = open_dir(indexdir)
    parser = MultifieldParser(
        ["titre", "contenu"], ix.schema, group=OrGroup.factory(0.9)
    )
    query = parser.parse(" ".join(terms))
    with ix.searcher() as searcher:
        results = searcher.search(query)
        for r in results:
            print(
                f'https://ville.sainte-adele.qc.ca/upload/documents/{r["document"]}#page={r["page"]} {r["titre"]}'
            )
