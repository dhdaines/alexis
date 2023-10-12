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


if __name__ == "__main__":
    test_format_html()
