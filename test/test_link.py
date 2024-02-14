import pytest

from alexi.link import Resolver

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
    "doc": {
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
        "index.html#1314-2021-Z",
    ),
    (
        "Règlement sur les permis et certificats 1314-2021-PC",
        "index.html#1314-2021-PC",
    ),
]


@pytest.mark.parametrize("test_input,expected", BYLAWS)
def test_bylaws(test_input, expected):
    r = Resolver(METADATA)
    assert r.resolve_internal(test_input) == expected
