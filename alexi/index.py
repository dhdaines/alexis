"""
Construire un index pour faire des recherches dans les données extraites.
"""

import logging
import re
from pathlib import Path

from whoosh.analysis import CharsetFilter, StemmingAnalyzer  # type: ignore
from whoosh.fields import ID, NUMERIC, TEXT, Schema  # type: ignore
from whoosh.index import create_in  # type: ignore
from whoosh.support.charset import charset_table_to_dict  # type: ignore
from whoosh.support.charset import default_charset

LOGGER = logging.getLogger("index")
CHARMAP = charset_table_to_dict(default_charset)
ANALYZER = StemmingAnalyzer() | CharsetFilter(CHARMAP)


def index(indir: Path, outdir: Path):
    outdir.mkdir(exist_ok=True)
    schema = Schema(
        document=ID(stored=True),
        page=NUMERIC(stored=True),
        titre=TEXT(ANALYZER, stored=True),
        contenu=TEXT(ANALYZER, stored=True),
    )
    ix = create_in(outdir, schema)
    writer = ix.writer()
    for docdir in indir.iterdir():
        if not docdir.is_dir():
            continue
        LOGGER.info("Indexing %s", docdir.name)
        with open(docdir / "index.md") as infh:
            titre = docdir.name
            contenu = infh.read()
            if m := re.search(r"^# (.*)", contenu, re.M):
                titre = m.group(1)
            writer.add_document(
                document=docdir.name, page=1, titre=titre, contenu=contenu
            )
        articles = docdir / "Article"
        if articles.is_dir():
            for artdir in articles.iterdir():
                LOGGER.info("Indexing article %s", artdir.name)
                with open(artdir / "index.md") as infh:
                    titre = artdir.name
                    contenu = infh.read()
                    # First header is title
                    if m := re.search(r"^#+ (.*)", contenu, re.M):
                        titre = m.group(1)
                    writer.add_document(
                        document=docdir.name, page=1, titre=titre, contenu=contenu
                    )
    writer.commit()
