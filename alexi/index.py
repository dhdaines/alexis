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


"""
    for path in jsons:
        reg = Reglement.parse_file(path)
        for texte in reg.textes:
            titre = [f"Règlement {reg.numero}"]
            if hasattr(texte, "article"):
                titre.append(f"Article {texte.article}")
            elif hasattr(texte, "annexe"):
                titre.append(f"Annexe {texte.annexe}")
            if texte.titre is not None:
                titre.append(texte.titre)
            writer.add_document(
                document=str(reg.fichier),
                page=texte.pages[0],
                titre=" ".join(titre),
                contenu="\n\n".join(c.texte for c in texte.contenu),
            )
    writer.commit()
"""
