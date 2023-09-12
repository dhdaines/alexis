#!/usr/bin/env python3

from typing import Any, Optional

from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfdevice import PDFDevice, PDFTextSeq
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.pdfinterp import (PDFGraphicState, PDFPageInterpreter,
                                PDFResourceManager, PDFStackT, PDFTextState)
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjRef
from pdfminer.psparser import PSLiteral
from pdfminer.utils import Matrix, make_compat_bytes


class MyTagExtractor(PDFDevice):
    """Extraire les tags pour populer la structure d'un document"""

    pageno: int

    def __init__(self, rsrcmgr: PDFResourceManager):
        super().__init__(rsrcmgr)
        self.pageno = 1

    def render_string(
        self,
        textstate: PDFTextState,
        seq: PDFTextSeq,
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState,
    ) -> None:
        font = textstate.font
        assert font is not None
        text = ""
        for obj in seq:
            if isinstance(obj, str):
                obj = make_compat_bytes(obj)
            if not isinstance(obj, bytes):  # Not sure what this means!
                continue
            chars = font.decode(obj)
            for cid in chars:
                try:
                    char = font.to_unichr(cid)
                    text += char
                except PDFUnicodeNotDefined:
                    pass
        print("TEXT", text)

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        pass

    def end_page(self, page: PDFPage) -> None:
        self.pageno += 1

    def begin_tag(self, tag: PSLiteral, props: Optional[PDFStackT] = None) -> None:
        print("BEGIN", tag.name, props)

    def end_tag(self) -> None:
        print("END")

    def do_tag(self, tag: PSLiteral, props: Optional["PDFStackT"] = None) -> None:
        print("TAG", tag.name, props)


def resolve_all(x: object, seen: Optional[set] = None) -> Any:
    if seen is None:
        seen = set()
    else:
        if repr(x) in seen:
            return x
        seen.add(repr(x))
    while isinstance(x, PDFObjRef):
        x = x.resolve()
    if isinstance(x, list):
        x = [resolve_all(v, seen=seen) for v in x]
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = resolve_all(v, seen=seen)
    return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="fichier PDF")
    args = parser.parse_args()
    with open(
        args.pdf,
        "rb",
    ) as fh:
        doc = PDFDocument(PDFParser(fh))
        struct_tree_root = resolve_all(doc.catalog["StructTreeRoot"])
        print(struct_tree_root)

    with open(
        args.pdf,
        "rb",
    ) as fh:
        rm = PDFResourceManager()
        device = MyTagExtractor(rm)
        interpreter = PDFPageInterpreter(rm, device)
        for page in PDFPage.get_pages(fh):
            interpreter.process_page(page)
