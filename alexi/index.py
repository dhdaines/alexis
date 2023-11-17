"""
Construire un index pour faire des recherches dans les données extraites.
"""

import json
import logging
import os
from pathlib import Path

from whoosh.analysis import CharsetFilter, StemmingAnalyzer  # type: ignore
from whoosh.fields import ID, NUMERIC, TEXT, Schema  # type: ignore
from whoosh.index import create_in  # type: ignore
from whoosh.support.charset import charset_table_to_dict  # type: ignore
from whoosh.support.charset import default_charset

LOGGER = logging.getLogger("index")
CHARMAP = charset_table_to_dict(default_charset)
ANALYZER = StemmingAnalyzer() | CharsetFilter(CHARMAP)


def add_from_dir(writer, document, docdir):
    LOGGER.info("Indexing %s", docdir)
    with open(docdir / "index.json") as infh:
        element = json.load(infh)
        titre = f'{element["type"]} {element["numero"]}: {element["titre"]}'
        page = element.get("page", 1)
    with open(docdir / "index.md") as infh:
        writer.add_document(
            document=document, page=page, titre=titre, contenu=infh.read()
        )


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
        document = docdir.with_suffix(".pdf").name
        add_from_dir(writer, document, docdir)
        for subdir in docdir.iterdir():
            if not docdir.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(subdir):
                if "index.json" in filenames:
                    add_from_dir(writer, document, Path(dirpath))
    writer.commit()
