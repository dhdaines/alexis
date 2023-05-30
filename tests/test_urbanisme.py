"""
Valider l'extraction des règlements d'urbanisme.
"""

import json
from pathlib import Path

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
    "\nCHAPITRE 3 DISPOSITIONS RELATIVES AUX ABC\n",
    "\nSECTION 1 DISPOSITIONS GÉNÉRALES\nSOUS-SECTION 1.1 OBTENTION ET ÉMISSION D'UN ABC\n7. Nécessité d’un ABC\nUn ABC est requis pour toute opération.\n",
    "\nSOUS-SECTION 1.2 DEMANDE DE PERMIS DE ABC\nSUR DEF GHI\n8. Documents additionnels requis\n",
    "\nENTRÉE EN VIGUEUR\nLe présent règlement entre en vigueur conformément à la loi.\nAvis de motion 17 mai 2021\nAdoption 19 juillet 2021\nEntrée en vigueur 23 septembre 2021\nSigné à Sainte-Adèle, ce 14e jour du mois d’octobre de l’an 2021.\n",
    "ANNEXE A ABC\n",
    "ANNEXE B XYZ\n",
    "X\nY\nZ\n",
]


def test_extract_urbanisme():
    """Valider l'extraction d'un règlement d'urbanisme à partir du texte."""
    ex = Extracteur("xyz-abc.pdf")
    reg = ex.extract_text(PAGES)
    print(reg.json(indent=2, ensure_ascii=False))
    assert reg.fichier == "xyz-abc.pdf"
    assert reg.objet == "XYZ ABC foo bar"
    assert reg.dates.avis == "17 mai 2021"
    assert reg.dates.adoption == "19 juillet 2021"
    assert reg.dates.entree == "23 septembre 2021"
    assert len(reg.chapitres) == 3
    assert reg.chapitres[0].titre == "DISPOSITIONS DÉCLARATOIRES,\nINTERPRÉTATIVES ET ADMINISTRATIVES\n"
    assert len(reg.chapitres[0].sections) == 4
    assert reg.chapitres[0].articles == (0, 5)
    assert reg.chapitres[0].pages == (3, 8)
    assert len(reg.chapitres[1].sections) == 1
    assert reg.chapitres[1].pages == (8, 10)
    assert reg.chapitres[2].pages == (10, 14)
    assert len(reg.chapitres[2].sections) == 1
    assert len(reg.chapitres[2].sections[0].sous_sections) == 2
    assert reg.chapitres[2].sections[0].sous_sections[0].articles == (6, 7)
    assert reg.chapitres[2].sections[0].sous_sections[1].articles == (7, 8)
    assert len(reg.annexes) == 2
    assert reg.annexes[0].titre == "ABC\n"
    assert reg.annexes[0].numero == "A"
    assert reg.annexes[1].titre == "XYZ\n"
    assert reg.annexes[1].numero == "B"

