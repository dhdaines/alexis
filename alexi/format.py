"""
Formatter la structure extraite d'un PDF
"""

import logging
from pathlib import Path
from typing import Optional

from alexi.analyse import Bloc, Document, Element

LOGGER = logging.getLogger("format")


def format_xml(doc: Document, indent: int = 2) -> str:
    """Représentation structurel du document."""

    def bloc_xml(bloc: Bloc) -> str:
        return f"<{bloc.type}>{bloc.texte}</{bloc.type}>"

    def element_xml(el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        spacing = " " * offset
        lines = [spacing + f"<{el.palier} titre='{el.titre}'>"]
        idx = el.debut
        fin = len(doc.blocs) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(element_xml(sub, indent, offset + indent))
                idx = len(doc.blocs) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                subspacing = " " * (offset + indent)
                lines.append(subspacing + bloc_xml(doc.blocs[idx]))
                idx += 1
        lines.append(spacing + f"</{el.palier}>")
        return lines

    return "\n".join(element_xml(doc.root, indent))


TAG = {
    "Document": "body",
    "Chapitre": "section",
    "Section": "section",
    "SousSection": "section",
    "Article": "article",
    "Annexe": "section",
}
HEADER = {
    "Document": "h1",
    "Chapitre": "h1",
    "Section": "h2",
    "SousSection": "h3",
    "Article": "h4",
    "Annexe": "h1",
}
BLOC = {
    "Tete": "",
    "Pied": "",
    "TOC": "",
    "Tableau": "",
    "Figure": "",
    "Liste": "li",
    "Titre": "h4",
    "Alinea": "p",
    "Amendement": "p",
}


def format_html(doc: Document, pdf: Optional[Path] = None, indent: int = 2) -> str:
    """Représentation HTML5 du document."""

    def bloc_html(bloc: Bloc) -> str:
        if pdf and bloc.type in ("Tableau", "Figure"):
            return ""
        tag = BLOC[bloc.type]
        if tag == "":
            return ""
        return f"<{tag}>{bloc.texte}</{tag}>"

    def element_html(el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        spacing = " " * offset
        subspacing = " " * (offset + indent)
        tag = TAG[el.palier]
        header = HEADER[el.palier]
        lines = [spacing + f"<{tag}>"]
        if el.titre:
            lines.append(subspacing + f"<{header}>{el.titre}</{header}>")
        idx = el.debut
        fin = len(doc.blocs) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(element_html(sub, indent, offset + indent))
                idx = len(doc.blocs) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                html = bloc_html(doc.blocs[idx])
                if html:
                    lines.append(subspacing + html)
                idx += 1
        lines.append(spacing + f"</{tag}>")
        return lines

    return "\n".join(element_html(doc.root, indent))
