import json
from pathlib import Path

import pytest

from alexi.analyse import Document, match_links
from alexi.link import Resolver, locate_article

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
    (
        "Code de la sécurité routière (L.R.Q., c. C-24.2)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/C-24.2",
    ),
    (
        "Règlement sur la signalisation routière (R.R.Q., c. C-24, r.28)",
        "https://www.legisquebec.gouv.qc.ca/fr/document/rc/C-24,%20r.%2028%20",
    ),
    (
        "Loi sur l’aménagement et l’urbanisme",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/A-19.1",
    ),
    (
        "Loi sur la qualité de l'environnement",
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/Q-2",
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
    # common enough that we need to deal with it
    (
        "chapitre 6 du Règlement de zonage",
        "../20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/6/index.html",
    ),
    (
        "Règlement de zonage",
        "../index.html#20231213-Codification-administrative-Rgl-1314-2021-Z",
    ),
    # NOTE: sous-section not expected to work yet
    (
        "section 3 du chapitre 5 du Règlement de zonage 1314-2021-Z",
        "../20231213-Codification-administrative-Rgl-1314-2021-Z/Chapitre/5/Section/3/index.html",
    ),
    # TODO: links to milieux, usages, etc
    # FIXME: Also test invalid links somehow!
]


@pytest.mark.parametrize("test_input,expected", BYLAWS)
def test_bylaws(test_input, expected):
    r = Resolver(METADATA)
    assert r.resolve_internal(test_input, ".") == expected


with open(DATADIR / "lotissement.json", "rt") as infh:
    LOTISSEMENT = Document.fromdict(json.load(infh))
with open(DATADIR / "zonage.json", "rt") as infh:
    ZONAGE = Document.fromdict(json.load(infh))

INTERNALS = [
    ("article 5", "../5/index.html", "Article/6", None),
    ("chapitre 2", "../../Chapitre/2/index.html", "Article/6", None),
    (
        "section 2 du chapitre 3",
        "../../Chapitre/3/Section/2/index.html",
        "Article/6",
        None,
    ),
    ("section 3", "../3/index.html", "Chapitre/3/Section/2", None),
    ("chapitre 1", "../../../1/index.html", "Chapitre/3/Section/2", None),
    ("section 1", "Section/1/index.html", "Chapitre/3", None),
    ("article 7", "../../../../Article/7/index.html", "Chapitre/3/Section/2", None),
    (
        "section 1",
        "../../Chapitre/3/Section/1/index.html",
        "Article/69",  # Is in Chapitre 3 Section 2
        LOTISSEMENT,
    ),
    (
        "section 3 du présent chapitre",
        "../../Chapitre/1/Section/3/index.html",
        "Article/1",  # Is in Chapitre 1 Section 1
        LOTISSEMENT,
    ),
    (
        "section 3 du présent chapitre",
        "../../Chapitre/1/Section/3/index.html",
        "Article/1",  # Is in Chapitre 1 Section 1
        LOTISSEMENT,
    ),
    (
        "Section 3 du Chapitre 4",
        "../../Chapitre/4/Section/3/index.html",
        "Article/70",
        None,
    ),
    (
        "article 99",
        None,  # This article does not exit
        "Article/75",
        LOTISSEMENT,
    ),
    (
        "section 3 du chapitre 3",
        None,  # This section does not exit
        "Article/15",
        LOTISSEMENT,
    ),
    (
        "Sous-section 3.6 de la Section 3 du présent chapitre",
        "../../Chapitre/4/Section/3/SousSection/_34/index.html",
        "Article/233",
        ZONAGE,
    ),
]


@pytest.mark.parametrize("test_input,expected,sourcepath,doc", INTERNALS)
def test_internal_links(test_input, expected, sourcepath, doc):
    r = Resolver(METADATA)
    assert r.resolve_internal(test_input, sourcepath, doc) == expected


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


@pytest.mark.parametrize("test_input,expected,sourcepath,doc", INTERNALS)
def test_match_internals(test_input, expected, sourcepath, doc):
    """Verifier qu'on peut reconnaitre les sections"""
    links = list(match_links(test_input))
    assert links
    assert links[0].start <= 2  # l'
    if "présent" not in test_input:
        assert links[0].end == len(test_input)


LOCATE = [
    ("233", "Chapitre/4/Section/5/SousSection/_40"),
    ("69", "Chapitre/3/Section/4/SousSection/_13"),
]


@pytest.mark.parametrize("test_input,expected", LOCATE)
def test_locate_article(test_input, expected):
    """Verifier le placement des articles dans l'hierarchie"""
    path = "/".join(locate_article(test_input, ZONAGE))
    assert path == expected


MULTIPLES = [
    (
        "articles 227, 229 et 231 de la Loi sur l’aménagement et l’urbanisme (LRQ, A-19.1)",
        (
            "articles",
            ["227", "229", "231"],
            "Loi sur l’aménagement et l’urbanisme (LRQ, A-19.1)",
        ),
    ),
    (
        "articles 256.1, 256.2 ou 256.3 de la Loi sur l’aménagement et l’urbanisme (L.R.Q., chapitre A-19.1)",
        (
            "articles",
            ["256.1", "256.2", "256.3"],
            "Loi sur l’aménagement et l’urbanisme (L.R.Q., chapitre A-19.1)",
        ),
    ),
    (
        "articles 148.0.8 et 148.0.9 de la Loi sur l’aménagement et l’urbanisme (LRQ A-19.1)",
        (
            "articles",
            ["148.0.8", "148.0.9"],
            "Loi sur l’aménagement et l’urbanisme (LRQ A-19.1)",
        ),
    ),
    (
        "types des milieux T5.1, T5.2, T5.3, ZC.1 et ZC.2 du Règlement de zonage 1314-2021-Z",
        (
            "types des milieux",
            ["T5.1", "T5.2", "T5.3", "ZC.1", "ZC.2"],
            "Règlement de zonage 1314-2021-Z",
        ),
    ),
]


@pytest.mark.parametrize("text,_", MULTIPLES)
def test_match_multiples(text, _):
    """Verifier qu'on peut reconnaitre les sections multiples"""
    links = list(match_links(text))
    assert links
    assert links[0].start <= 2  # l'
