import csv
from pathlib import Path

from alexi.analyse import Analyseur
from alexi.format import format_html

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data"


def test_format_html():
    with open(TRAINDIR / "zonage_sections.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("zonage_sections", reader)
        doc = analyseur()
        html = format_html(doc)
        assert html.count("<h1") == 2
        assert html.count("<h2") == 4
        assert html.count("<h3") == 3
        assert html.count("<h4") == 26

        html = format_html(doc, element=doc.paliers["SousSection"][0])
        assert html.count("<h3") == 1
        assert html.count("<h4") == 6


def test_format_html_figures():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("pdf_structure", reader)
        doc = analyseur()
        html = format_html(doc)
        print(html)


if __name__ == "__main__":
    test_format_html()
    test_format_html_figures()
