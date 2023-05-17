"""
Construire un index pour faire des recherches dans les données extraites.
"""

import argparse
from pathlib import Path
from typing import List

from whoosh.analysis import CharsetFilter, StemmingAnalyzer  # type: ignore
from whoosh.fields import NUMERIC, TEXT, Schema  # type: ignore
from whoosh.index import create_in  # type: ignore
from whoosh.support.charset import charset_table_to_dict, default_charset  # type: ignore

from alexis.models import Reglement

CHARMAP = charset_table_to_dict(default_charset)
ANALYZER = StemmingAnalyzer() | CharsetFilter(CHARMAP)


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--outdir",
        help="Repertoire destination pour l'index",
        type=Path,
        default="indexdir",
    )
    parser.add_argument("jsons", help="Fichiers JSON", type=Path, nargs="+")
    return parser


def index(outdir: Path, jsons: List[Path]):
    outdir.mkdir(exist_ok=True)
    schema = Schema(
        sequence=NUMERIC(stored=True),
        page=NUMERIC,
        titre=TEXT(ANALYZER),
        contenu=TEXT(ANALYZER),
    )
    ix = create_in(outdir, schema)
    writer = ix.writer()
    for path in jsons:
        reg = Reglement.parse_file(path)
        for idx, article in enumerate(reg.articles):
            writer.add_document(
                sequence=idx,
                page=article.page,
                titre=article.titre,
                contenu="\n".join(article.alineas),
            )
    writer.commit()


def main():
    parser = make_argparse()
    args = parser.parse_args()
    index(args.outdir, args.jsons)

if __name__ == "__main__":
    main()
