import csv
from pathlib import Path

from alexi.analyse import Analyseur, extract_zonage, group_iob
from alexi.recognize import Objets

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
    with open(TRAINDIR / "patches" / "zonage_sections.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("zonage_sections", reader)
        doc = analyseur()
        assert doc.titre.strip() == "Règlement de zonage"
        assert doc.numero == "1314-2021-Z"
        assert doc.fileid == "zonage_sections"
        assert doc.structure.type == "Document"
        assert len(doc.structure.sub) == 1
        assert doc.structure.sub[0].type == "Chapitre"
        assert doc.structure.sub[0].numero == "2"
        assert (
            doc.structure.sub[0].titre
            == "DISPOSITIONS DÉCLARATOIRES, INTERPRÉTATIVES ET ADMINISTRATIVES"
        )
        assert doc.structure.sub[0].sub[0].type == "Section"
        assert doc.structure.sub[0].sub[0].sub[0].type == "Article"


def test_analyse_tableaux_figures():
    obj = Objets()
    with open(DATADIR / "pdf_figures.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("pdf_figures", reader)
        analyseur.add_images(obj(DATADIR / "pdf_figures.pdf"))
        doc = analyseur()
        assert "Figure" in (bloc.type for bloc in doc.contenu)
        assert "Tableau" in (bloc.type for bloc in doc.contenu)


ZONES = {
    "categorie_milieu": {
        "T1": {"titre": "MILIEUX NATURELS", "url": "zonage_zones/Chapitre/7/Section/2"},
        "T2": {
            "titre": "OCCUPATION DE LA FORÊT",
            "url": "zonage_zones/Chapitre/7/Section/3",
        },
    },
    "milieu": {
        "T1.1": {
            "titre": "CONSERVATION",
            "url": "zonage_zones/Chapitre/7/Section/2/SousSection/_2",
        },
        "T1.2": {
            "titre": "Récréation",
            "url": "zonage_zones/Chapitre/7/Section/2/SousSection/_3",
        },
        "T2.1": {
            "titre": "AGROFORESTIER",
            "url": "zonage_zones/Chapitre/7/Section/3/SousSection/_4",
        },
        "T2.2": {
            "titre": "RÉCRÉOTOURISTIQUE EXTENSIF",
            "url": "zonage_zones/Chapitre/7/Section/3/SousSection/_5",
        },
    },
}


def test_analyse_zonage():
    with open(TRAINDIR / "patches" / "zonage_zones.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        analyseur = Analyseur("zonage_zones", reader)
        doc = analyseur()
        zones = extract_zonage(doc)
        assert zones == ZONES


if __name__ == "__main__":
    test_iob()
    test_analyse()
    test_analyse_tableaux_figures()
