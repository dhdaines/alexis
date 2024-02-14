"""
Extraction de hyperliens du texte des règlements
"""

import logging
import re
from typing import Optional

LOGGER = logging.getLogger("link")
LQ_RE = re.compile(r"\(R?LRQ[^\)]+(?P<lq>[A-Z]- ?[\d\.]+)\)")
SEC_RE = re.compile(
    r"\b(?P<sec>article|chapitre|section|sous-section|annexe) (?P<num>[\d\.]+)"
)
REG_RE = re.compile(r"règlement[^\d]+(?P<reg>[\d\.A-Z-]+)", re.IGNORECASE)
DOT_RE = re.compile(r"\.")


class Resolver:
    def __init__(self, metadata: Optional[dict] = None):
        self.metadata = {"doc": {}} if metadata is None else metadata
        self.docpath = {}
        for docpath, info in self.metadata["doc"].items():
            self.docpath[info["numero"]] = docpath

    def resolve_internal(self, text: str) -> Optional[str]:
        """
        Resoudre certains liens internes.
        """
        if m := REG_RE.search(text):
            numero = m.group("reg")
            if numero is None:
                return None
            docpath = self.docpath.get(m.group("reg"))
            if docpath is None:
                return None
        secpath = []
        for m in SEC_RE.finditer(text):
            sectype = m.group("sec")
            num = m.group("num")
            secpath.append((sectype, num))
        if secpath:
            # FIXME
            return None
        else:
            return f"index.html#{numero}"

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
                num = DOT_RE.sub("_", num)
                url += f"#se:{num}"
                break
        return url
