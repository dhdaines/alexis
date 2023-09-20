import csv
from pathlib import Path

from alexi.analyse import group_iob, Analyseur

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data" / "train"

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
        for element in group_iob(reader):
            tagged.append(element.xml)
    assert tagged == IOBTEST


def test_analyse():
    with open(TRAINDIR / "zonage_sections.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur()
        doc = analyseur(reader)
        assert doc.xml.count("<Chapitre") == 1
        assert doc.xml.count("<Section") == 4
        assert doc.xml.count("<SousSection") == 3


if __name__ == "__main__":
    test_iob()
    test_analyse()
