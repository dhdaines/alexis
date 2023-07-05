"""
Valider l'extraction des règlements d'urbanisme.
"""

import json
from alexi.extraction.urbanisme import Extracteur

PAGES = [
    "Règlement sur XYZ ABC foo bar\nNuméro 1314-2021-XYZABC",
    "Règlement sur XYZ ABC foo bar numéro 1314-2021-XYZABC Table des matières, des figures et des tableaux\nTable des matières\nCHAPITRE 1 DISPOSITIONS DÉCLARATOIRES, INTERPRÉTATIVES ET ADMINISTRATIVES\n.................................................................................................................................. 8\nCHAPITRE 2 DISPOSITIONS GÉNÉRALES AUX XYZ ABC ..................................................................................................................16\n",
    "Règlement sur XYZ ABC 1314-2021-XYZABC Table des matières, des figures et des tableaux\nTable des tableaux\nTableau 1 – Montant d'une amende en fonction du type d'infraction ................................................................................... 14\n",
    "\nCHAPITRE 1 DISPOSITIONS DÉCLARATOIRES,\nINTERPRÉTATIVES ET ADMINISTRATIVES\n",
    "\nSECTION 1 DISPOSITIONS DÉCLARATOIRES\n1. Titre du règlement\nLe présent règlement est intitulé « Règlement sur XYZ ABC et porte le numéro\n1314-2021-XYZABC.\n2. Abrogation\nLe présent règlement abroge le règlement numéro 1200-2012-XYZABC « Règlement relatif aux XYZ ABC\nd’autorisation » tel que modifié par tous leurs amendements ainsi que toutes dispositions inconciliables d’un autre\nrèglement en vigueur.\n",
    "\nSECTION 2 DISPOSITIONS INTERPRÉTATIVES\n3. Ville\nL’expression « Ville » est définie comme étant la Ville de Sainte-Adèle.\n",
    "\nSECTION 3 DISPOSITIONS ADMINISTRATIVES\n4. Administration du règlement\nL'administration du présent règlement est confiée à toute personne nommée ci-après « fonctionnaire désigné », par\nrésolution du Conseil.\n",
    "\nSECTION 4 INFRACTIONS, PÉNALITÉS ET RECOURS\n5. Infraction\nCommets une infraction, quiconque ne se conforme pas à une disposition du présent règlement ainsi qu’à tout règlement\nen vigueur.\nSans restreindre la portée du premier alinéa commet une infraction, quiconque :\n1. ABC DEF XYZ.\n",
    "\nCHAPITRE 2 DISPOSITIONS GÉNÉRALES AUX XYZ\nET AUTRES\n",
    "\nSECTION 1 DISPOSITIONS GÉNÉRALES\n6. Procuration\nSi le requérant ou un mandataire d’un permis ou d’un certificat n’est pas le propriétaire de l’immeuble ou du bien meuble\nvisé par la demande\n",
    "\nSECTION 2 ABC – ABCDEF\nINTENTION\nLa « ABC ABCDEF », a pour objectif de ABC DEF XYZ\n7. Le ABC est DEF.\n",
    "\nCHAPITRE 3 DISPOSITIONS RELATIVES AUX ABC\n",
    "\nSECTION 1 DISPOSITIONS GÉNÉRALES\nSOUS-SECTION 1.1 OBTENTION ET ÉMISSION D'UN ABC\n8. Nécessité d’un ABC\nUn ABC est requis pour toute opération.\n",
    "\nSOUS-SECTION 1.2 DEMANDE DE PERMIS DE ABC\nSUR DEF GHI\n9. Documents additionnels requis\n",
    "\nENTRÉE EN VIGUEUR\nLe présent règlement entre en vigueur conformément à la loi.\nAvis de motion 17 mai 2021\nAdoption 19 juillet 2021\nEntrée en vigueur 23 septembre 2021\nSigné à Sainte-Adèle, ce 14e jour du mois d’octobre de l’an 2021.\n",
    "ANNEXE A ABC\n",
    "ANNEXE B XYZ\n",
    "X\nY\nZ\n",
]


OUTPUT = {
    "fichier": "xyz-abc.pdf",
    "numero": "1314-2021-XYZABC",
    "titre": "Règlement sur XYZ ABC foo bar Numéro 1314-2021-XYZABC",
    "objet": "XYZ ABC foo bar",
    "dates": {
        "adoption": "19 juillet 2021",
        "avis": "17 mai 2021",
        "entree": "23 septembre 2021",
    },
    "chapitres": [
        {
            "numero": "1",
            "titre": "DISPOSITIONS DÉCLARATOIRES, INTERPRÉTATIVES ET ADMINISTRATIVES",
            "pages": [3, 8],
            "articles": [0, 5],
            "sections": [
                {
                    "numero": "1",
                    "titre": "DISPOSITIONS DÉCLARATOIRES",
                    "pages": [4, 5],
                    "articles": [0, 2],
                    "sous_sections": [],
                },
                {
                    "numero": "2",
                    "titre": "DISPOSITIONS INTERPRÉTATIVES",
                    "pages": [5, 6],
                    "articles": [2, 3],
                    "sous_sections": [],
                },
                {
                    "numero": "3",
                    "titre": "DISPOSITIONS ADMINISTRATIVES",
                    "pages": [6, 7],
                    "articles": [3, 4],
                    "sous_sections": [],
                },
                {
                    "numero": "4",
                    "titre": "INFRACTIONS, PÉNALITÉS ET RECOURS",
                    "pages": [7, 8],
                    "articles": [4, 5],
                    "sous_sections": [],
                },
            ],
        },
        {
            "numero": "2",
            "titre": "DISPOSITIONS GÉNÉRALES AUX XYZ ET AUTRES",
            "pages": [8, 11],
            "articles": [5, 8],
            "sections": [
                {
                    "numero": "1",
                    "titre": "DISPOSITIONS GÉNÉRALES",
                    "pages": [9, 10],
                    "articles": [5, 6],
                    "sous_sections": [],
                },
                {
                    "numero": "2",
                    "titre": "ABC – ABCDEF",
                    "pages": [10, 11],
                    "articles": [6, 8],
                    "sous_sections": [],
                },
            ],
        },
        {
            "numero": "3",
            "titre": "DISPOSITIONS RELATIVES AUX ABC",
            "pages": [11, 15],
            "articles": [8, 11],
            "sections": [
                {
                    "numero": "1",
                    "titre": "DISPOSITIONS GÉNÉRALES",
                    "pages": [12, 15],
                    "articles": [8, 11],
                    "sous_sections": [
                        {
                            "numero": "1",
                            "titre": "OBTENTION ET ÉMISSION D'UN ABC",
                            "pages": [12, 13],
                            "articles": [8, 9],
                        },
                        {
                            "numero": "2",
                            "titre": "DEMANDE DE PERMIS DE ABC",
                            "pages": [13, 15],
                            "articles": [9, 11],
                        },
                    ],
                }
            ],
        },
    ],
    "articles": [
        {
            "titre": "Titre du règlement",
            "pages": [4, 4],
            "alineas": [
                "Le présent règlement est intitulé « Règlement sur XYZ ABC et porte le numéro",
                "1314-2021-XYZABC.",
            ],
            "numero": 1,
            "sous_section": -1,
            "section": 0,
            "chapitre": 0,
        },
        {
            "titre": "Abrogation",
            "pages": [4, 5],
            "alineas": [
                "Le présent règlement abroge le règlement numéro 1200-2012-XYZABC « Règlement relatif aux XYZ ABC",
                "d’autorisation » tel que modifié par tous leurs amendements ainsi que toutes dispositions inconciliables d’un autre",
                "règlement en vigueur.",
            ],
            "numero": 2,
            "sous_section": -1,
            "section": 0,
            "chapitre": 0,
        },
        {
            "titre": "Ville",
            "pages": [5, 6],
            "alineas": [
                "L’expression « Ville » est définie comme étant la Ville de Sainte-Adèle."
            ],
            "numero": 3,
            "sous_section": -1,
            "section": 1,
            "chapitre": 0,
        },
        {
            "titre": "Administration du règlement",
            "pages": [6, 7],
            "alineas": [
                "L'administration du présent règlement est confiée à toute personne nommée ci-après « fonctionnaire désigné », par",
                "résolution du Conseil.",
            ],
            "numero": 4,
            "sous_section": -1,
            "section": 2,
            "chapitre": 0,
        },
        {
            "titre": "Infraction",
            "pages": [7, 8],
            "alineas": [
                "Commets une infraction, quiconque ne se conforme pas à une disposition du présent règlement ainsi qu’à tout règlement",
                "en vigueur.",
                "Sans restreindre la portée du premier alinéa commet une infraction, quiconque :",
                "1. ABC DEF XYZ.",
            ],
            "numero": 5,
            "sous_section": -1,
            "section": 3,
            "chapitre": 0,
        },
        {
            "titre": "Procuration",
            "pages": [9, 10],
            "alineas": [
                "Si le requérant ou un mandataire d’un permis ou d’un certificat n’est pas le propriétaire de l’immeuble ou du bien meuble",
                "visé par la demande",
            ],
            "numero": 6,
            "sous_section": -1,
            "section": 0,
            "chapitre": 1,
        },
        {
            "titre": "INTENTION",
            "pages": [10, 10],
            "alineas": ["La « ABC ABCDEF », a pour objectif de ABC DEF XYZ"],
            "numero": -1,
            "sous_section": -1,
            "section": 1,
            "chapitre": 1,
        },
        {
            "titre": "Le ABC est DEF.",
            "pages": [10, 11],
            "alineas": [],
            "numero": 7,
            "sous_section": -1,
            "section": 1,
            "chapitre": 1,
        },
        {
            "titre": "Nécessité d’un ABC",
            "pages": [12, 13],
            "alineas": ["Un ABC est requis pour toute opération."],
            "numero": 8,
            "sous_section": 0,
            "section": 0,
            "chapitre": 2,
        },
        {
            "titre": "SUR DEF GHI",
            "pages": [13, 13],
            "alineas": [],
            "numero": -1,
            "sous_section": 1,
            "section": 0,
            "chapitre": 2,
        },
        {
            "titre": "Documents additionnels requis",
            "pages": [13, 15],
            "alineas": [""],
            "numero": 9,
            "sous_section": 1,
            "section": 0,
            "chapitre": 2,
        },
    ],
    "annexes": [
        {"titre": "ABC", "pages": [15, 16], "alineas": [], "numero": "A"},
        {
            "titre": "XYZ",
            "pages": [16, 18],
            "alineas": ["X", "Y", "Z"],
            "numero": "B",
        },
    ],
}


def test_extract_urbanisme():
    """Valider l'extraction d'un règlement d'urbanisme à partir du texte."""
    ex = Extracteur("xyz-abc.pdf")
    reg = ex.extract_text(PAGES)
    assert reg.fichier == "xyz-abc.pdf"
    assert reg.objet == "XYZ ABC foo bar"
    assert reg.dates.avis == "17 mai 2021"
    assert reg.dates.adoption == "19 juillet 2021"
    assert reg.dates.entree == "23 septembre 2021"
    assert len(reg.chapitres) == 3
    assert (
        reg.chapitres[0].titre
        == "DISPOSITIONS DÉCLARATOIRES, INTERPRÉTATIVES ET ADMINISTRATIVES"
    )
    assert len(reg.chapitres[0].sections) == 4
    assert reg.chapitres[0].articles == (0, 5)
    assert reg.chapitres[0].pages == (3, 8)
    assert len(reg.chapitres[1].sections) == 2
    assert reg.chapitres[1].pages == (8, 11)
    assert reg.chapitres[1].sections[1].articles == (6, 8)
    assert reg.articles[6].titre == "INTENTION"
    assert reg.articles[6].alineas == [
        "La « ABC ABCDEF », a pour objectif de ABC DEF XYZ",
    ]
    assert reg.chapitres[2].pages == (11, 15)
    assert len(reg.chapitres[2].sections) == 1
    assert len(reg.chapitres[2].sections[0].sous_sections) == 2
    assert reg.chapitres[2].sections[0].sous_sections[0].articles == (8, 9)
    assert reg.chapitres[2].sections[0].sous_sections[1].articles == (9, 11)
    assert len(reg.annexes) == 2
    assert reg.annexes[0].titre == "ABC"
    assert reg.annexes[0].numero == "A"
    assert reg.annexes[1].titre == "XYZ"
    assert reg.annexes[1].numero == "B"
    assert json.loads(reg.json()) == OUTPUT


if __name__ == "__main__":
    ex = Extracteur("xyz-abc.pdf")
    reg = ex.extract_text(PAGES)
    print(reg.json(indent=2, ensure_ascii=False))
