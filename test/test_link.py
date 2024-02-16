import json
from pathlib import Path

import pytest

from alexi.analyse import Document, match_links
from alexi.link import Resolver

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data"
LAWS = [
    (
        "Loi sur l’aménagement et l’urbanisme (LRQ, chapitre A-19.1)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/A-19.1",
    ),
    (
        "Loi sur la sécurité civile (LRQ, chapitre S-2.3)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/S-2.3",
    ),
    (
        "Loi sur le patrimoine culturel (LRQ P-9.002)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/P-9.002",
    ),
    (
        "l’article 148.0.20.1 de la Loi sur l’aménagement et l’urbanisme (LRQ A- 19.1)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/A-19.1#se:148_0_20_1",
    ),
    (
        "l’article 357 de la Loi sur les cités et villes (RLRQ, c. C-19)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-19#se:357",
    ),
    (
        "Règlement sur le captage des eaux souterraines (L.R.Q. c. Q-2, r. 6)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/rc/Q-2,%20r.%206%20",
    ),
    (
        "Loi sur la qualité de l'environnement (L.R.Q.,c. Q-2)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/Q-2",
    ),
    (
        "Règlement sur les normes d’intervention dans les forêts du domaine public (c. F-4.1, r. 7)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/rc/F-4.1,%20r.%207%20",
    ),
]


@pytest.mark.parametrize("test_input,expected", LAWS)
def test_laws(test_input, expected):
    r = Resolver()
    assert r.resolve_external(test_input) == expected


METADATA = {
    "zonage": {
        "categorie_milieu": {
            "T5": {
                "titre": "CENTRE-VILLE",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/6",
            },
        },
        "milieu": {
            "T5.1": {
                "titre": "VILLAGEOIS",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/6/SousSection/_88",
            },
            "T5.2": {
                "titre": "NÉO-VILLAGEOIS",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/6/SousSection/_89",
            },
            "T5.3": {
                "titre": "COMPACT",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/6/SousSection/_90",
            },
            "ZC.1": {
                "titre": "COMMERCE RÉCRÉOTOURISTIQUE INTENSIF",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/9/SousSection/_95",
            },
            "ZC.2": {
                "titre": "COMMERCIAL",
                "url": "20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/7/Section/9/SousSection/_96",
            },
        },
    },
    "docs": {
        "Rgl-1314-2021-PC-version-en-vigueur-20231013": {
            "numero": "1314-2021-PC",
            "titre": "Règlement sur les permis et certificats ",
            "pdf": "https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-PC-version-en-vigueur-20231013.pdf",
        },
        "20231213-Codification-administrative-Rgl-1314-2021-Z": {
            "numero": "1314-2021-Z",
            "titre": "Règlement de zonage ",
            "pdf": "https://ville.sainte-adele.qc.ca/upload/documents/20231213-Codification-administrative-Rgl-1314-2021-Z.pdf",
        },
    },
}
BYLAWS = [
    (
        "Règlement de zonage 1314-2021-Z",
        "../index.html#20231213-Codification-administrative-Rgl-1314-2021-Z",
    ),
    (
        "Règlement sur les permis et certificats 1314-2021-PC",
        "../index.html#Rgl-1314-2021-PC-version-en-vigueur-20231013",
    ),
    (
        "chapitre 5 du Règlement de zonage 1314-2021-Z",
        "../20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/5/index.html",
    ),
    # NOTE: sous-section not expected to work yet
    (
        "section 3 du chapitre 5 du Règlement de zonage 1314-2021-Z",
        "../20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/5/Section/3/index.html",
    ),
    # TODO: links to milieux, usages, etc
]


@pytest.mark.parametrize("test_input,expected", BYLAWS)
def test_bylaws(test_input, expected):
    r = Resolver(METADATA)
    assert r.resolve_internal(test_input, ".") == expected


INTERNALS = [
    (
        "article 5",
        "../5/index.html",
        "Article/6",
    ),
    (
        "chapitre 2",
        "../../Chapitre/2/index.html",
        "Article/6",
    ),
    (
        "section 2 du chapitre 3",
        "../../Chapitre/3/Section/2/index.html",
        "Article/6",
    ),
    (
        "section 3",
        "../3/index.html",
        "Chapitre/3/Section/2",
    ),
    (
        "chapitre 1",
        "../../../1/index.html",
        "Chapitre/3/Section/2",
    ),
    (
        "section 1",
        "Section/1/index.html",
        "Chapitre/3",
    ),
    (
        "article 7",
        "../../../../Article/7/index.html",
        "Chapitre/3/Section/2",
    ),
    (
        "section 1",
        "../../Chapitre/3/Section/1/index.html",
        "Article/69",  # Is in Chapitre 3 Section 2
    ),
    (
        "section 3 du présent chapitre",
        "../../Chapitre/1/Section/3/index.html",
        "Article/1",  # Is in Chapitre 1 Section 1
    ),
]

with open(DATADIR / "lotissement.json", "rt") as infh:
    LOTISSEMENT = Document.fromdict(json.load(infh))


@pytest.mark.parametrize("test_input,expected,sourcepath", INTERNALS)
def test_internal_links(test_input, expected, sourcepath):
    r = Resolver(METADATA)
    assert r.resolve_internal(test_input, sourcepath, LOTISSEMENT) == expected


@pytest.mark.parametrize("test_input,expected", LAWS)
def test_match_laws(test_input, expected):
    """Verifier qu'on peut reconnaitre les lois"""
    links = list(match_links(test_input))
    assert links
    assert links[0].start <= 2  # l'
    assert links[0].end == len(test_input)


@pytest.mark.parametrize("test_input,expected", BYLAWS)
def test_match_bylaws(test_input, expected):
    """Verifier qu'on peut reconnaitre les reglements"""
    links = list(match_links(test_input))
    assert links
    assert links[0].start <= 2  # l'
    assert links[0].end == len(test_input)


@pytest.mark.parametrize("test_input,expected,_", INTERNALS)
def test_match_internals(test_input, expected, _):
    """Verifier qu'on peut reconnaitre les sections"""
    links = list(match_links(test_input))
    assert links
    assert links[0].start <= 2  # l'
    if "présent" not in test_input:
        assert links[0].end == len(test_input)


MULTIPLES = [
    "articles 227, 229 et 231 de la Loi sur l’aménagement et l’urbanisme (LRQ, A-19.1)",
    "articles 148.0.8 et 148.0.9 de la Loi sur l’aménagement et l’urbanisme (LRQ A-19.1)",
    "types des milieux T5.1, T5.2, T5.3, ZC.1 et ZC.2 du Règlement de zonage 1314-2021-Z",
]


@pytest.mark.parametrize("text", MULTIPLES)
def test_match_multiples(text):
    """Verifier qu'on peut reconnaitre les sections multiples"""
    links = list(match_links(text))
    assert links
    assert links[0].start <= 2  # l'
