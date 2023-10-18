"""
Formatter la structure extraite d'un PDF
"""

import itertools
import logging
from typing import Iterator, Optional, Sequence

from alexi.analyse import Bloc, Document, Element, T_obj

LOGGER = logging.getLogger("format")


def line_breaks(paragraph: Sequence[T_obj]) -> Iterator[list[T_obj]]:
    if len(paragraph) == 0:
        return
    xdeltas = [int(paragraph[0]["x0"])]
    xdeltas.extend(
        int(b["x0"]) - int(a["x0"]) for a, b in itertools.pairwise(paragraph)
    )
    ydeltas = [int(paragraph[0]["top"])]
    ydeltas.extend(
        int(b["top"]) - int(a["top"]) for a, b in itertools.pairwise(paragraph)
    )
    line: list[T_obj] = []
    for word, xdelta, ydelta in zip(paragraph, xdeltas, ydeltas):
        if xdelta <= 0 and ydelta > 0:  # CR, LF
            yield line
            line = []
        line.append(word)
    if line:
        yield line


def format_xml(doc: Document, indent: int = 2) -> str:
    """Représentation structurel du document."""

    def bloc_xml(bloc: Bloc) -> str:
        return f"<{bloc.type}>{bloc.texte}</{bloc.type}>"

    def element_xml(el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        spacing = " " * offset
        lines = [spacing + f"<{el.type} titre='{el.titre}'>"]
        idx = el.debut
        fin = len(doc.contenu) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(element_xml(sub, indent, offset + indent))
                idx = len(doc.contenu) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                subspacing = " " * (offset + indent)
                lines.append(subspacing + bloc_xml(doc.contenu[idx]))
                idx += 1
        lines.append(spacing + f"</{el.type}>")
        return lines

    return "\n".join(element_xml(doc.structure, indent))


TAG = {
    "Document": "body",
    "Annexe": "section",
    "Chapitre": "section",
    "Section": "section",
    "SousSection": "section",
    "Article": "article",
}
HEADER = {
    "Document": "h1",
    "Annexe": "h1",
    "Chapitre": "h1",
    "Section": "h2",
    "SousSection": "h3",
    "Article": "h4",
}
BLOC = {
    "Tete": "",
    "Pied": "",
    "TOC": "",
    "Tableau": "img",
    "Figure": "img",
    "Liste": "li",
    "Titre": "h4",
    "Alinea": "p",
    "Amendement": "p",
}


def format_html(
    doc: Document,
    indent: int = 2,
    element: Optional[Element] = None,
) -> str:
    """Représentation HTML5 du document."""

    def bloc_html(bloc: Bloc) -> str:
        tag = BLOC[bloc.type]
        if tag == "":
            return ""
        elif tag == "img":
            return f'<img alt="{bloc.texte}" src="{bloc.img}">'
        else:
            return f"<{tag}>{bloc.texte}</{tag}>"

    def element_html(el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        spacing = " " * offset
        subspacing = " " * (offset + indent)
        tag = TAG[el.type]
        header = HEADER[el.type]
        lines = [spacing + f"<{tag}>"]
        if el.titre:
            lines.append(subspacing + f"<{header}>{el.titre}</{header}>")
        idx = el.debut
        fin = len(doc.contenu) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(element_html(sub, indent, offset + indent))
                idx = len(doc.contenu) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                html = bloc_html(doc.contenu[idx])
                if html:
                    lines.append(subspacing + html)
                idx += 1
        lines.append(spacing + f"</{tag}>")
        return lines

    if element is None:
        element = doc.structure
    return "\n".join(element_html(element, indent))


def format_text(
    doc: Document,
    element: Optional[Element] = None,
) -> str:
    """Contenu textuel du document."""

    def bloc_text(bloc: Bloc) -> str:
        tag = BLOC[bloc.type]
        if tag == "":
            return ""
        return "\n".join(
            " ".join(w["text"] for w in line) for line in line_breaks(bloc.contenu)
        )

    def element_text(el: Element) -> list[str]:
        lines = [el.titre, "-" * len(el.titre), ""]
        idx = el.debut
        fin = len(doc.contenu) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(element_text(sub))
                idx = len(doc.contenu) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                lines.append(bloc_text(doc.contenu[idx]))
                lines.append("")
                idx += 1
        lines.append("")
        return lines

    if element is None:
        element = doc.structure
    return "\n".join(element_text(element))
