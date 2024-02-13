"""
Formatter la structure extraite d'un PDF
"""

import itertools
import logging
from pathlib import Path
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
    imgdir: str = ".",
    fragment: bool = True,
) -> str:
    """Représentation HTML5 du document."""
    imgpath = Path(imgdir)

    def bloc_html(bloc: Bloc) -> str:
        tag = BLOC[bloc.type]
        if tag == "":
            return ""
        elif tag == "img":
            return f'<img alt="{bloc.texte}" src="{imgpath / bloc.img}"><br>'
        else:
            return f"<{tag}>{bloc.texte}</{tag}>"

    def element_html(el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        off = " " * offset
        sp = " " * indent
        tag = TAG[el.type]
        header = HEADER[el.type]
        lines = []
        if tag != "body":
            lines.append(f'{off}<{tag} class="{el.type}">')
        if el.numero and offset:
            lines.append(
                f'{off}{sp}<a class="anchor" name="{el.type}/{el.numero}"></a>'
            )
        if el.titre:
            lines.append(f'{off}{sp}<{header} class="header">')
            if el.type != "Document":
                if el.numero[0] != "_":
                    lines.append(f'{off}{sp}{sp}<span class="level">{el.type}</span>')
                    lines.append(
                        f'{off}{sp}{sp}<span class="number">{el.numero}</span>'
                    )
            lines.append(f'{off}{sp}{sp}<span class="title">{el.titre}</span>')
            lines.append(f"{off}{sp}</{header}>")
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
                    lines.append(off + sp + html)
                idx += 1
        if tag != "body":
            lines.append(off + f"</{tag}>")
        return lines

    if element is None:
        doc_body = "\n".join(element_html(doc.structure, indent))
    else:
        doc_body = "\n".join(element_html(element, indent))
    if fragment:
        return doc_body
    else:
        doc_header = f"""<!DOCTYPE html>
<html>
  <head>
    <title>{doc.titre}</title>
  </head>
  <body>"""
        doc_footer = "</body></html>"
        return "\n".join((doc_header, doc_body, doc_footer))
