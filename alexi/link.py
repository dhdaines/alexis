"""
Extraction de hyperliens du texte des règlements
"""

import itertools
import logging
import os
import re
from collections import deque
from typing import Optional, Iterator

from .analyse import PALIERS, Document, Element

LOGGER = logging.getLogger("link")

# FIXME: Synchronize with analyse regexps
LQ_RE = re.compile(
    r"\(\s*(?:c\.|(?:R\.?\s*)?[LR]\.?\s*R\.?\s*Q\.?)\s*,"
    r"?(?:\s*(?:c(?:\.|\s+)|chapitre\s+))?(?P<lq>[^\)]+)\)"
)
RQ_RE = re.compile(r"(?P<lq>.*?),\s*r.\s*(?P<rq>.*)")
SEC_RE = re.compile(
    r"\b(?P<sec>article|chapitre|section|sous-section|annexe) (?P<num>[\d\.]+)",
    re.IGNORECASE,
)
REG_RE = re.compile(r"règlement[^\d]+(?P<reg>[\d\.A-Z-]+)", re.IGNORECASE)
PALIER_IDX = {palier: idx for idx, palier in enumerate(PALIERS)}


def locate_article(numero: str, doc: Document) -> list[str]:
    """
    Placer un article dans l'hierarchie du document.
    """
    for path, el in doc.structure.traverse():
        if el.type == "Article" and el.numero == numero:
            return path


def normalize_title(title: str):
    title = title.lower()
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^règlement (?:de|sur|concernant) ", "", title)
    title = re.sub(r"\([^\)]+\)$", "", title)
    return title


class Resolver:
    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = {"docs": {}} if metadata is None else metadata
        self.numeros = {}
        self.titles = {}
        for docpath, info in self.metadata["docs"].items():
            self.numeros[info["numero"]] = docpath
            self.titles[normalize_title(info["titre"])] = docpath

    def __call__(
        self, text: str, srcpath: str = "", doc: Optional[Document] = None
    ) -> str:
        url = self.resolve_external(text)
        if url:
            return url
        return self.resolve_internal(text, srcpath, doc)

    def resolve_absolute_internal(
        self, docpath: str, secpath: str, srcpath: str
    ) -> Optional[str]:
        if secpath:
            return os.path.relpath(f"../{docpath}/{secpath}/index.html", srcpath)
        else:
            return os.path.relpath(f"../index.html#{docpath}", srcpath)

    def resolve_internal(
        self, text: str, srcpath: str, doc: Optional[Document] = None
    ) -> Optional[str]:
        """
        Resoudre certains liens internes.
        """
        docpath = None
        text = re.sub(r"\s+", " ", text).strip()
        # NOTE: This really matches anything starting with "règlement"
        if m := REG_RE.search(text):
            numero = m.group("reg").strip(" .,;")
            if numero is None:
                return None
            docpath = self.numeros.get(numero)
            if docpath is None:
                for title in self.titles:
                    if title in text.lower():
                        docpath = self.titles[title]
                        break
            if docpath is None:
                return None

        sections = []
        for m in SEC_RE.finditer(text):
            sectype = m.group("sec").title().replace("-", "")
            num = m.group("num").strip(" .,;")
            sections.append((sectype, num))
        sections.sort(key=lambda x: PALIER_IDX.get(x[0], 0))
        secpath = "/".join(itertools.chain.from_iterable(sections))
        if docpath:
            return self.resolve_absolute_internal(docpath, secpath, srcpath)
        if not secpath:
            return None

        srcparts = list(srcpath.split("/"))
        secparts = self.qualify_destination(list(secpath.split("/")), srcparts, doc)
        secpath = self.resolve_document_path(secparts, doc)
        relpath = os.path.relpath("/".join(secparts), srcpath)
        href = f"{relpath}/index.html"
        LOGGER.info("resolve %s à partir de %s: %s", secpath, srcpath, href)
        return href

    def qualify_destination(
        self, dest: list[str], src: list[str], doc: Optional[Document]
    ) -> list[str]:
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

    def resolve_document_path(
        self, dest: list[str], doc: Optional[Document] = None
    ) -> Optional[str]:
        """Verifier la présence d'une cible de lien dans un document et retourner le path."""
        if doc is None:
            return "/".join(dest)
        # Resolve by path with fuzzing for missing section/chapter/subsection numbers

        # Resolve by title if possible (NOTE: not currently possible)

        return "/".join(dest)

    def resolve_external(self, text: str) -> Optional[str]:
        """
        Resoudre quelques types de liens externes (vers la LAU par exemple)
        """
        text = re.sub(r"\s+", " ", text).strip()
        if m := LQ_RE.search(text):
            lq = m.group("lq").strip()
            if m := RQ_RE.match(lq):
                # Format the super wacky URL style for reglements
                lq = m.group("lq")
                rq = m.group("rq")
                reg = f"{lq},%20r.%20{rq}%20"
                url = f"https://www.legisquebec.gouv.qc.ca/fr/document/rc/{reg}"
            else:
                loi = re.sub(r"\s+", "", lq)
                url = f"https://www.legisquebec.gouv.qc.ca/fr/document/lc/{loi}"
        elif "code civil" in text.lower():
            url = "https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991"
        elif "cités et villes" in text.lower():
            url = "https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-19"
        elif "urbanisme" in text.lower():
            url = "https://www.legisquebec.gouv.qc.ca/fr/document/lc/A-19.1"
        elif "environnement" in text.lower():
            url = "https://www.legisquebec.gouv.qc.ca/fr/document/lc/Q-2"
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
