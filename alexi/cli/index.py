"""
Construire un index pour faire des recherches dans les données extraites.
"""

from pathlib import Path
from typing import List

from alexi.types import Reglement
from whoosh.analysis import CharsetFilter, StemmingAnalyzer  # type: ignore
from whoosh.fields import ID, NUMERIC, TEXT, Schema  # type: ignore
from whoosh.index import create_in  # type: ignore
from whoosh.support.charset import charset_table_to_dict  # type: ignore
from whoosh.support.charset import default_charset

CHARMAP = charset_table_to_dict(default_charset)
ANALYZER = StemmingAnalyzer() | CharsetFilter(CHARMAP)


def index(outdir: Path, jsons: List[Path]):
    outdir.mkdir(exist_ok=True)
    schema = Schema(
        document=ID(stored=True),
        page=NUMERIC(stored=True),
        titre=TEXT(ANALYZER, stored=True),
        contenu=TEXT(ANALYZER, stored=True),
    )
    ix = create_in(outdir, schema)
    writer = ix.writer()
    for path in jsons:
        reg = Reglement.parse_file(path)
        for article in reg.articles:
            writer.add_document(
                document=reg.fichier,
                page=article.pages[0],
                titre=f"Règlement {reg.numero} Article {article.numero}\n{article.titre}",
                contenu="\n".join(article.alineas),
            )
        for chapitre in reg.chapitres:
            for section in chapitre.sections:
                writer.add_document(
                    document=reg.fichier,
                    page=section.pages[0],
                    titre=f"Règlement {reg.numero} Section {chapitre.numero}.{section.numero}\n{section.titre}",
                    contenu="\n\n".join(f"{article.numero}. {article.titre}\n" + "\n".join(article.alineas)
                                        for article in reg.articles[section.articles[0]:section.articles[1]]))
        for annexe in reg.annexes:
            writer.add_document(
                document=reg.fichier,
                page=annexe.pages[0],
                titre=f"Règlement {reg.numero} Annexe {annexe.numero}\n{annexe.titre}",
                contenu="\n".join(annexe.alineas))
    writer.commit()


def main(args):
    index(args.outdir, args.jsons)
