"""
Formatter la structure extraite d'un PDF
"""

from collections import deque
import itertools
import logging
import re
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
            return f'<img alt="{bloc.texte}" src="{bloc.img}"><br>'
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

    if element is not None:
        return "\n".join(element_html(element, indent))
    doc_body = "\n".join(element_html(doc.structure, indent))
    doc_header = f"""<!DOCTYPE html>
<html>
  <head>
    <title>{doc.meta.get("Titre", "Document")}</title>
  </head>"""
    doc_footer = "</html>"
    return "\n".join((doc_header, doc_body, doc_footer))


MDHEADER = {
    "Document": "#",
    "Annexe": "#",
    "Chapitre": "#",
    "Section": "##",
    "SousSection": "###",
    "Article": "####",
}


def format_text(
    doc: Document,
    element: Optional[Element] = None,
) -> str:
    """Contenu textuel du document."""

    def bloc_text(bloc: Bloc) -> str:
        tag = BLOC[bloc.type]
        if tag in ("", "img"):
            return ""
        return "\n".join(
            " ".join(w["text"] for w in line) for line in line_breaks(bloc.contenu)
        )

    def element_text(el: Element) -> list[str]:
        """Générer du texte (en fait du Markdown) d'un élément."""
        header = MDHEADER[el.type]
        lines = []
        if el.titre:
            lines.append(" ".join((header, el.titre)))
            lines.append("")
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
                txt = bloc_text(doc.contenu[idx])
                if txt:
                    lines.append(txt)
                    lines.append("")
                idx += 1
        lines.append("")
        return lines

    if element is None:
        element = doc.structure
    return "\n".join(element_text(element))


def format_dict(doc: Document, imgdir: str = ".") -> str:  # noqa: C901
    """Formatter un document en dictionnaire afin d'émettre un JSON pour
    utilisation dans SÈRAFIM"""
    # structure de base
    doc_dict = {
        "fichier": None,
        "titre": doc.meta.get("Titre"),
        "numero": doc.meta.get("Numero"),
        "chapitres": [],
        "textes": [],
        "dates": {
            "adoption": None,
        },
    }
    if doc_dict.get("numero") is None and doc_dict.get("titre") is not None:
        tokens = doc_dict["titre"].split()
        for t in tokens:
            if re.match(r".*\d\d", t) and re.match(r"^[0-9A-Z-]+$", t):
                LOGGER.info("Quelque chose qui ressemble à un numéro: %s", t)
                doc_dict["numero"] = t

    def bloc_dict(bloc: Bloc) -> dict:
        tag = BLOC[bloc.type]
        if tag == "":
            return {}
        elif tag == "img":
            imgpath = "/".join((imgdir, bloc.img))
            if bloc.type == "Tableau":
                return {"texte": bloc.texte, "tableau": imgpath}
            else:
                return {"texte": bloc.texte, "figure": imgpath}
        else:
            return {"texte": bloc.texte}

    # group together "contenu" as "texte" (they are not the same thing)
    def make_texte(titre: str, contenus: Sequence[Bloc], page: int) -> dict:
        contenu = []
        for bloc in contenus:
            bd = bloc_dict(bloc)
            if bd:
                contenu.append(bd)
        if len(contenus) == 0:
            pages = [page, page]
        else:
            pages = ([int(contenus[0].page_number), int(contenus[-1].page_number)],)
        texte = {
            "titre": titre,
            "pages": pages,
            "contenu": contenu,
        }
        if m := re.match(r"(?:article )?(\d+)", titre, re.I):
            texte["article"] = int(m.group(1))
        return texte

    # add front matter as a single texte
    if not doc.structure.sub:
        LOGGER.warning("Absence de structure dans le document!")
        preambule = doc.contenu
    else:
        preambule = doc.contenu[0 : doc.structure.sub[0].debut]
    pretexte = make_texte(doc.meta.get("Titre", "Préambule"), preambule, 1)
    if pretexte:
        doc_dict["textes"].append(pretexte)

    # depth-first traverse adding leaf nodes as textes. total hack,
    # not refactored, doomed to go away at some point
    d = deque(doc.structure.sub)
    chapitre = None
    chapitre_idx = 0
    section = None
    section_idx = 0
    sous_section = None
    sous_section_idx = 0
    while d:
        el = d.popleft()
        if el.sub:
            if el.type == "Chapitre":
                chapitre_idx += 1
                if m := re.match(r"(?:chapitre )?(\d+|[XIV]+)", el.titre, re.I):
                    chapitre_numero = m.group(1)
                    el.titre = el.titre[m.end(1) :]
                else:
                    chapitre_numero = "%d" % chapitre_idx
                first_page = int(doc.contenu[el.debut].page_number)
                end = len(doc.contenu) if el.fin == -1 else el.fin
                last_page = int(doc.contenu[end - 1].page_number)
                if chapitre:
                    chapitre["textes"][1] = len(doc_dict["textes"])
                if section:
                    section["textes"][1] = len(doc_dict["textes"])
                if sous_section:
                    sous_section["textes"][1] = len(doc_dict["textes"])
                chapitre = {
                    "numero": chapitre_numero,
                    "titre": el.titre,
                    "pages": [first_page, last_page],
                    "textes": [len(doc_dict["textes"]), -1],
                }
                doc_dict["chapitres"].append(chapitre)
                section_idx = 0
                sous_section_idx = 0
            elif el.type == "Section":
                section_idx += 1
                if m := re.match(r"(?:section )?(\d+|[XIV]+)", el.titre, re.I):
                    section_numero = m.group(1)
                    el.titre = el.titre[m.end(1) :]
                else:
                    section_numero = "%d" % section_idx
                first_page = int(doc.contenu[el.debut].page_number)
                end = len(doc.contenu) if el.fin == -1 else el.fin
                last_page = int(doc.contenu[end - 1].page_number)
                if section:
                    section["textes"][1] = len(doc_dict["textes"])
                if sous_section:
                    sous_section["textes"][1] = len(doc_dict["textes"])
                section = {
                    "numero": section_numero,
                    "titre": el.titre,
                    "pages": [first_page, last_page],
                    "textes": [len(doc_dict["textes"]), -1],
                }
                if chapitre:
                    chapitre.setdefault("sections", []).append(section)
            elif el.type == "SousSection":
                sous_section_idx += 1
                if m := re.match(r"(?:sous-section )?([\d\.]+)", el.titre, re.I):
                    sous_section_numero = m.group(1)
                    el.titre = el.titre[m.end(1) :]
                else:
                    sous_section_numero = "%d" % sous_section_idx
                first_page = int(doc.contenu[el.debut].page_number)
                end = len(doc.contenu) if el.fin == -1 else el.fin
                last_page = int(doc.contenu[end - 1].page_number)
                if sous_section:
                    sous_section["textes"][1] = len(doc_dict["textes"])
                sous_section = {
                    "numero": sous_section_numero,
                    "titre": el.titre,
                    "pages": [first_page, last_page],
                    "textes": [len(doc_dict["textes"]), -1],
                }
                if section:
                    section.setdefault("sous_sections", []).append(sous_section)
            d.extendleft(reversed(el.sub))
        else:
            if el.type == "Annexe":
                if m := re.match(r"(?:annexe )?(\d+|[A-Z]\b)", el.titre, re.I):
                    annexe = m.group(1)
                    el.titre = el.titre[m.end(1) :]
                else:
                    annexe = "A"
            start = el.debut
            end = len(doc.contenu) if el.fin == -1 else el.fin
            texte = make_texte(el.titre, doc.contenu[start:end], el.page)
            if not texte:
                continue
            if el.type == "Annexe":
                texte["annexe"] = annexe
                if chapitre:
                    chapitre["textes"][1] = len(doc_dict["textes"])
                    chapitre = None
                if section:
                    section["textes"][1] = len(doc_dict["textes"])
                    section = None
                if sous_section:
                    sous_section["textes"][1] = len(doc_dict["textes"])
                    sous_section = None
            else:
                if chapitre:
                    texte["chapitre"] = chapitre_idx - 1
                if section:
                    texte["section"] = section_idx - 1
                if sous_section:
                    texte["sous_section"] = sous_section_idx - 1
                doc_dict["textes"].append(texte)
    # FIXME: not actually correct, but we don't really care
    if chapitre and chapitre["textes"][1] == -1:
        chapitre["textes"][1] = len(doc_dict["textes"])
    if section and section["textes"][1] == -1:
        section["textes"][1] = len(doc_dict["textes"])
    if sous_section and sous_section["textes"][1] == -1:
        sous_section["textes"][1] = len(doc_dict["textes"])

    return doc_dict
