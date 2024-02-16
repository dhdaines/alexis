import csv
import re
from pathlib import Path

from alexi.analyse import Analyseur, Hyperlien
from alexi.format import HtmlFormatter, format_html

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


HTML = """<h1 class="header">
    <span class="title">Document</span>
  </h1>
  <h4>Titre du document</h4>
  <h4>Titre 1</h4>
  <p>Contenu 1, contenu 2, contenu 3.</p>
  <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
  <h4>Titre 2</h4>
  <p>Encore du contenu!</p>
  <li>1. Énumération 1</li>
  <li>2. Énumération 2</li>
  <li>a) Énumération imbriquée</li>
  <p>3. Longue énumération : Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
  <p>Tableau</p>
  <p>Chose Truc</p>
  <p>Chose 1 Truc 1</p>
  <p>Chose 2 Truc 2</p>"""


def test_format_html_figures():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("pdf_structure", reader)
        doc = analyseur()
        html = format_html(doc)
        assert re.sub(r"\s+", " ", html).strip() == re.sub(r"\s+", " ", HTML).strip()


LIPSUM = "<p>Lorem ipsum dolor sit amet"
LIPSUM_HREF = (
    '<p>Lorem <a target="_blank" href="https://example.com">ipsum</a>'
    ' dolor <a target="_blank" href="https://vdsa.ca">sit</a> amet'
)


def test_format_html_links():
    """Tester la génération de liens."""
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("pdf_structure", reader)
        doc = analyseur()
        formatter = HtmlFormatter()
        assert formatter.bloc_html(doc.contenu[3]).startswith(LIPSUM)
        doc.contenu[3].liens = [
            Hyperlien("https://example.com", 6, 11),
            Hyperlien("https://vdsa.ca", 18, 21),
        ]
        assert formatter.bloc_html(doc.contenu[3]).startswith(LIPSUM_HREF)


if __name__ == "__main__":
    test_format_html()
    test_format_html_figures()
