"""
Extraction de hyperliens du texte des règlements
"""

import itertools
import logging
import os
import re
from collections import deque
from typing import Optional

from .analyse import PALIERS, Document

LOGGER = logging.getLogger("link")
LQ_RE = re.compile(r"\(R?LRQ[^\)]+(?P<lq>[A-Z]- ?[\d\.]+)\)")
SEC_RE = re.compile(
    r"\b(?P<sec>article|chapitre|section|sous-section|annexe) (?P<num>[\d\.]+)"
)
REG_RE = re.compile(r"règlement[^\d]+(?P<reg>[\d\.A-Z-]+)", re.IGNORECASE)
PALIER_IDX = {palier: idx for idx, palier in enumerate(PALIERS)}


def locate_article(numero: str, doc: Document) -> list[str]:
    """
    Placer un article dans l'hierarchie du document.
    """
    d = deque(doc.structure.sub)
    path = []
    while d:
        el = d.popleft()
        if el is None:
            path.pop()
            path.pop()
        elif el.sub:
            path.append(el.type)
            path.append(el.numero)
            d.appendleft(None)
            d.extendleft(reversed(el.sub))
        else:
            if el.type == "Article" and el.numero == numero:
                return path
    return []


def qualify_destination(dest: list[str], src: list[str], doc: Document) -> list[str]:
    """
    Rajouter des prefix manquants pour un lien relatif.
    """
    # Top-level section types
    if dest[0] in ("Chapitre", "Article", "Annexe"):
        return dest
    # Only fully qualified destinations are possible
    if src[0] == "Annexe":
        return dest
    # Need to identify enclosing section/subsection (generally these
    # are of the form "section N du présent chaptire"...).  Note that
    # we do not modify the source path here, so we will always end up
    # with a full destination path
    if src[0] in ("Article"):
        if doc is None or len(src) == 1:  # pathological situation...
            return dest
        src = locate_article(src[1], doc)
    try:
        idx = src.index(dest[0])
    except ValueError:
        idx = len(src)
    return src[:idx] + dest


def _resolve_internal(
    secpath: str, srcpath: str, doc: Optional[Document] = None
) -> str:
    secparts = list(secpath.split("/"))
    srcparts = list(srcpath.split("/"))
    secparts = qualify_destination(secparts, srcparts, doc)
    return os.path.relpath("/".join(secparts), "/".join(srcparts))


class Resolver:
    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = {"doc": {}} if metadata is None else metadata
        self.docpath = {}
        for docpath, info in self.metadata["doc"].items():
            self.docpath[info["numero"]] = docpath

    def resolve_absolute_internal(self, numero: str, secpath: str) -> Optional[str]:
        docpath = self.docpath.get(numero)
        if docpath is None:
            return None
        if secpath:
            return f"{docpath}/{secpath}/index.html"
        else:
            return f"index.html#{docpath}"

    def resolve_internal(
        self, text: str, srcpath: str, doc: Optional[Document] = None
    ) -> Optional[str]:
        """
        Resoudre certains liens internes.
        """
        numero = None
        if m := REG_RE.search(text):
            numero = m.group("reg")
            if numero is None:
                return None
        sections = []
        for m in SEC_RE.finditer(text):
            sectype = m.group("sec").title().replace("-", "")
            num = m.group("num")
            sections.append((sectype.title(), num))
        sections.sort(key=lambda x: PALIER_IDX.get(x[0], 0))
        secpath = "/".join(itertools.chain.from_iterable(sections))
        if numero:
            return self.resolve_absolute_internal(numero, secpath)
        return _resolve_internal(secpath, srcpath, doc)

    def resolve_external(self, text: str) -> Optional[str]:
        """
        Resoudre quelques types de liens externes (vers la LAU par exemple)
        """
        if m := LQ_RE.search(text):
            loi = re.sub(r"\s+", "", m.group("lq"))
            url = f"https://www.legisquebec.gouv.qc.ca/fr/document/lc/{loi}"
        elif "code civil" in text.lower():
            url = "https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991"
        else:
            return None
        for m in SEC_RE.finditer(text):
            sectype = m.group("sec")
            num = m.group("num")
            if sectype == "article":
                num = num.replace(".", "_")
                url += f"#se:{num}"
                break
        return url
