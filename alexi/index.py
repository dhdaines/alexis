"""
Construire un index pour faire des recherches dans les données extraites.
"""

from pathlib import Path
from typing import List

from whoosh.analysis import CharsetFilter, StemmingAnalyzer  # type: ignore
from whoosh.fields import ID, NUMERIC, TEXT, Schema  # type: ignore
from whoosh.index import create_in  # type: ignore
from whoosh.support.charset import charset_table_to_dict  # type: ignore
from whoosh.support.charset import default_charset

from alexi.types import Reglement

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
        for contenu in reg.contenus:
            titre = [f"Règlement {reg.numero}"]
            if hasattr(contenu, "article"):
                titre.append(f"Article {contenu.article}")
            if hasattr(contenu, "annexe"):
                titre.append(f"Annexe {contenu.annexe}")
            if contenu.titre is not None:
                titre.append(contenu.titre)
            writer.add_document(
                document=str(reg.fichier),
                page=contenu.pages[0],
                titre=" ".join(titre),
                contenu="\n\n".join(contenu.alineas),
            )
        for chapitre in reg.chapitres:
            for section in chapitre.sections:

                def make_contenu_texte(c):
                    alineas = []
                    if c.article:
                        alineas.append(f"{c.article}. {c.titre}\n")
                    return "\n\n".join(alineas)

                writer.add_document(
                    document=reg.fichier,
                    page=section.pages[0],
                    titre=f"Règlement {reg.numero} Section {chapitre.numero}.{section.numero}\n{section.titre}",
                    contenu="\n\n".join(
                        make_contenu_texte(c)
                        for c in reg.contenus[
                            section.contenus[0] : section.contenus[1] + 1
                        ]
                    ),
                )
    writer.commit()
