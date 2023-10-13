import csv
from pathlib import Path

from alexi.analyse import Analyseur
from alexi.format import format_html

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data" / "train"


def test_format_html():
    with open(TRAINDIR / "zonage_sections.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur()
        doc = analyseur(reader)
        html = format_html(doc)
        assert html.count("<h1") == 1
        assert html.count("<h2") == 4
        assert html.count("<h3") == 3
        assert html.count("<h4") == 26

        html = format_html(doc, element=doc.paliers["SousSection"][0])
        assert html.count("<h3") == 1
        assert html.count("<h4") == 6


def test_format_html_figures():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur()
        doc = analyseur(reader)
        html = format_html(doc, pdf=(DATADIR / "pdf_structure.pdf"))
        print(html)


if __name__ == "__main__":
    test_format_html()
    test_format_html_figures()
