"""Extraire la structure et le contenu d'un règlement d'urbanisme en
format PDF.
"""

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pdfplumber
import tqdm

from alexis import models


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Fichier PDF", type=Path)
    return parser


def clean_text(text: str) -> str:
    """Enlever en-tête, en-pied, et autres anomalies d'une page."""
    # En-pied
    text = re.sub("\n\\d+$", "\n", text)
    # En-tête (methode imparfaite...)
    text = re.sub("^\s*Règlement.*(Chapitre|Entrée en vigueur).*", "", text)
    return text


def extract_title(pages: List[str]) -> Tuple[Optional[str], Optional[str]]:
    numero, objet = None, None
    for page in pages:
        m = re.search(r"(?i:règlement)(?:\s+(?:de|d'|sur))?\s+(.*)\s+(?i:numéro\s+(\S+))", page)
        if m:
            objet = m.group(1)
            numero = m.group(2)
            break
    return numero, objet


def extract_dates(pages: List[str]) -> models.Dates:
    """Extraire les dates d'avis de motion, adoption, et entrée en vigueur
    d'un texte règlement."""
    dates = {}
    for i, page in enumerate(pages):
        m = re.search(r"^ENTRÉE EN VIGUEUR", page, re.MULTILINE)
        if not m:
            continue
        startpos = m.start(0)
        m = re.search(r"avis de motion (.*)$", page, re.MULTILINE | re.IGNORECASE)
        if m:
            dates["avis"] = m.group(1)
        m = re.search(r"adoption (.*)$", page, re.MULTILINE | re.IGNORECASE)
        if m:
            dates["adoption"] = m.group(1)
        m = re.search(r"entrée en vigueur (.*)$", page, re.MULTILINE | re.IGNORECASE)
        if m:
            dates["entree"] = m.group(1)
        # Si on est rendu ici, on les a eues pour de vrai, alors enlever
        # le texte de la page
        if "adoption" in dates and "entree" in dates:
            pages[i] = page[:startpos]
            break
    return models.Dates(**dates)

def extract_chapter(page: str, idx: int) -> Optional[models.Chapitre]:
    m = re.search(r"CHAPITRE\s+(\d+)\s+([^\.\d]+)(\d+)?$", page)
    if m:
        numero = m.group(1)
        return models.Chapitre(numero=numero, titre=m.group(2), page=idx)
    return None


def extract_annex(page: str, idx: int) -> Optional[models.Annexe]:
    m = re.search(r"ANNEXE\s+(\S+)(?: –)?\s+([^\.\d]+)(\d+)?$", page)
    if m:
        numero = m.group(1)
        return models.Annexe(numero=numero, titre=m.group(2), page=idx)
    return None


def extract_bylaw(pdf: Path) -> models.Reglement:
    """Extraire la structure d'un règlement d'urbanisme."""
    # D'abord extraire les elements (textes, tableaux)
    pages = []
    with pdfplumber.open(pdf) as doc:
        for page in tqdm.tqdm(doc.pages):
            texte = page.extract_text()
            texte = clean_text(texte)
            pages.append(texte)
    numero, objet = extract_title(pages)
    dates = extract_dates(pages)
    chapitres: List[models.Chapitre] = []
    articles: List[models.Article] = []
    annexes: List[models.Annexe] = []
    in_tables = True
    art_num = 0
    for i, texte in enumerate(pages):
        # Premier chapitre termine les tables de contenus
        chapter = extract_chapter(texte, i)
        if chapter is not None:
            chapitres.append(chapter)
            in_tables = False
            continue
        if in_tables:
            continue
        annex = extract_annex(texte, i)
        if annex is not None:
            annexes.append(annex)
        for line in texte.split("\n"):
            m = re.match(r"SECTION\s+(\d+)\s+(.*)", line)
            if m:
                sec = m.group(1)
                section = models.Section(numero=sec, titre=m.group(2), page=i)
                assert chapitres
                chapitres[-1].sections.append(section)
                continue
            m = re.match(r"SOUS-SECTION\s+\d+\.(\d+)\s+(.*)", line)
            if m:
                sec = m.group(1)
                sous_section = models.SousSection(numero=sec, titre=m.group(2), page=i)
                assert chapitres
                chapitres[-1].sections[-1].sous_sections.append(sous_section)
                continue
            if annexes:
                annexes[-1].alineas.append(line)
                continue
            m = re.match(r"(\d+)\.\s+(.*)", line)
            if m:
                num = int(m.group(1))
                if num > art_num:
                    art_num = num
                    assert chapitres
                    if chapitres[-1].debut == 0:
                        chapitres[-1].debut = art_num
                    if chapitres[-1].sections and chapitres[-1].sections[-1].debut == 0:
                        chapitres[-1].sections[-1].debut = art_num
                    if chapitres[-1].sections[-1].sous_sections and chapitres[-1].sections[-1].sous_sections[-1].debut == 0:
                        chapitres[-1].sections[-1].sous_sections[-1].debut = art_num
                    chapitres[-1].fin = art_num
                    if chapitres[-1].sections:
                        chapitres[-1].sections[-1].fin = art_num
                    if chapitres[-1].sections[-1].sous_sections:
                        chapitres[-1].sections[-1].sous_sections[-1].fin = art_num
                    articles.append(
                        models.Article(
                            numero=art_num, titre=m.group(2), page=i, alineas=[]
                        )
                    )
                else:
                    articles[-1].alineas.append(line)
                continue
            if articles:
                articles[-1].alineas.append(line)
    if numero is None:
        numero = "INCONNU"
    return models.Reglement(
        numero=numero, objet=objet, dates=dates, chapitres=chapitres, articles=articles, annexes=annexes
    )


def main():
    parser = make_argparse()
    args = parser.parse_args()
    print(extract_bylaw(args.pdf).json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
