"""
Formatter la structure extraite d'un PDF
"""

import itertools
import logging
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional, Sequence

from alexi.analyse import Bloc, Document, Element, T_obj
from alexi.link import Resolver

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

CURPATH = Path(".")


class HtmlFormatter:
    def __init__(
        self,
        doc: Document,
        imgdir: PathLike = CURPATH,
        resolver: Optional[Resolver] = None,
        path: Optional[Path] = None,
        indent: int = 2,
    ):
        self.imgpath = Path(imgdir)
        self.resolver = resolver
        self.path = path
        self.doc = doc
        self.indent = indent

    def bloc_html(self, bloc: Bloc) -> str:
        tag = BLOC[bloc.type]
        if tag == "":
            return ""
        elif tag == "img":
            return f'<img alt="{bloc.texte}" src="{self.imgpath / bloc.img}"><br>'
        elif bloc.liens:
            text = bloc.texte
            start = 0
            chunks = []
            for link in bloc.liens:
                chunks.append(text[start : link.start])
                link_text = text[link.start : link.end]
                href = link.href
                if href is None and self.resolver:
                    href_text = link_text if link.alt is None else link.alt
                    href = self.resolver(href_text, str(self.path), self.doc)
                    LOGGER.info("%s:%s -> %s", href_text, self.path, href)
                if href is None:
                    chunks.append(link_text)
                else:
                    chunks.append(f'<a target="_blank" href="{href}">{link_text}</a>')
                start = link.end
            chunks.append(text[start:])
            html = "".join(chunks)
            return f"<{tag}>{html}</{tag}>"
        else:
            return f"<{tag}>{bloc.texte}</{tag}>"

    def element_html(self, el: Element, indent: int = 2, offset: int = 0) -> list[str]:
        off = " " * offset
        sp = " " * indent
        tag = TAG[el.type]
        header = HEADER[el.type]
        lines = []
        LOGGER.debug("%s%s %s: %d-%d", off, el.type, el.numero, el.debut, el.fin)
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
            pdflink = self.pdflink(el)
            if pdflink:
                lines.append(f"{off}{sp}{sp}{pdflink}")
            lines.append(f"{off}{sp}</{header}>")
        idx = el.debut
        fin = len(self.doc.contenu) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(self.element_html(sub, indent, offset + indent))
                idx = len(self.doc.contenu) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                html = self.bloc_html(self.doc.contenu[idx])
                if html:
                    lines.append(off + sp + html)
                idx += 1
            LOGGER.debug("%s%d", off, len(lines))
        if tag != "body":
            lines.append(off + f"</{tag}>")
        return lines

    def pdflink(self, el: Optional[Element] = None) -> str:
        """Lien HTML vers la page du PDF original"""
        if self.doc.pdfurl is None:
            return ""
        if el is not None:
            return (
                f' (<a target="_blank" href="{self.doc.pdfurl}#page={el.page}">PDF</a>)'
            )
        else:
            return f' (<a target="_blank" href="{self.doc.pdfurl}">PDF</a>)'

    def __call__(
        self,
        element: Optional[Element] = None,
        fragment: bool = True,
    ) -> str:
        """Représentation HTML5 du document."""

        if element is None:
            lines = self.element_html(self.doc.structure, self.indent)
        else:
            lines = self.element_html(element, self.indent)
        LOGGER.debug("%d lignes", len(lines))
        doc_body = "\n".join(lines)
        if fragment:
            return doc_body
        else:
            doc_header = f"""<!DOCTYPE html>
    <html>
      <head>
        <title>{self.doc.titre}{self.pdflink()}</title>
      </head>
      <body>"""
            doc_footer = "</body></html>"
            return "\n".join((doc_header, doc_body, doc_footer))


def format_html(
    doc: Document,
    indent: int = 2,
    element: Optional[Element] = None,
    imgdir: str = ".",
    fragment: bool = True,
) -> str:
    return HtmlFormatter(imgdir=Path(imgdir), doc=doc, indent=indent)(element, fragment)
