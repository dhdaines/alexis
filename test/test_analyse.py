import csv
from pathlib import Path

from alexi.analyse import Analyseur, group_iob
from alexi.convert import Converteur

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data"

IOBTEST = [
    "<Titre>Titre incomplet</Titre>",
    "<Titre>Titre 1</Titre>",
    "<Alinea>Contenu</Alinea>",
    "<Alinea>Contenu</Alinea>",
    "<Alinea>incomplet incorrect</Alinea>",
    "<Alinea>Lorem ipsum dolor</Alinea>",
]


def test_iob():
    with open(DATADIR / "iob_test.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        tagged = []
        for bloc in group_iob(reader):
            tagged.append(f"<{bloc.type}>{bloc.texte}</{bloc.type}>")
    assert tagged == IOBTEST


def test_analyse():
    with open(TRAINDIR / "zonage_sections.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("zonage_sections", reader)
        doc = analyseur()


def test_analyse_tableaux_figures():
    conv = Converteur(DATADIR / "pdf_figures.pdf")
    with open(DATADIR / "pdf_figures.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("pdf_figures", reader)
        analyseur.add_images(conv.extract_images())
        doc = analyseur()
        assert "Figure" in (bloc.type for bloc in doc.contenu)
        assert "Tableau" in (bloc.type for bloc in doc.contenu)


if __name__ == "__main__":
    test_iob()
    test_analyse()
    test_analyse_tableaux_figures()
