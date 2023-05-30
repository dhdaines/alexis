"""Extraire la structure et le contenu d'un règlement d'urbanisme en
format PDF.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import pdfplumber
import tqdm

from alexi import models


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Extraire le texte seulement", action="store_true")
    parser.add_argument("pdf", help="Fichier PDF", type=Path)
    return parser


def clean_text(text: str) -> str:
    """Enlever en-tête, en-pied, et autres anomalies d'une page."""
    # En-pied
    text = re.sub("\n\\d+$", "\n", text)
    # En-tête (methode imparfaite...)
    text = re.sub(r"^\s*Règlement.*(Chapitre|Entrée en vigueur).*", "", text)
    return text


def extract_title(pages: List[str]) -> Tuple[Optional[str], Optional[str]]:
    numero, objet = None, None
    for page in pages:
        m = re.search(
            r"(?i:règlement)(?:\s+(?:de|d'|sur))?\s+(.*)\s+(?i:numéro\s+(\S+))", page
        )
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
        # le texte de la page (FIXME: means the last article has bogus pages)
        if "adoption" in dates and "entree" in dates:
            pages[i] = page[:startpos]
            break
    return models.Dates(**dates)


def extract_chapter(page: str, idx: int) -> Optional[models.Chapitre]:
    return None


def extract_text_from_pdf(pdf: Path) -> List[str]:
    """Extraire les pages d'un PDF."""
    pages = []
    with pdfplumber.open(pdf) as doc:
        for page in tqdm.tqdm(doc.pages):
            texte = page.extract_text()
            texte = clean_text(texte)
            pages.append(texte)
    return pages


class Extracteur:
    fichier: str = "INCONNU"
    numero: str = "INCONNU"
    objet: str = "INCONNU"
    dates: models.Dates
    chapitre: Optional[models.Chapitre] = None
    section: Optional[models.Section] = None
    sous_section: Optional[models.SousSection] = None
    annexe: Optional[models.Annexe] = None
    artidx: int = -1
    pageidx: int = -1

    def __init__(self, fichier: str = None):
        if fichier is not None:
            self.fichier = fichier
        self.pages: List[str] = []
        self.chapitres: List[models.Chapitre] = []
        self.articles: List[models.Article] = []
        self.annexes: List[models.Annexe] = []

    def close_annexe(self):
        """Clore la derniere annexe (et chapitre, et section, etc)"""
        if self.annexe:
            self.annexe.pages = (self.annexe.pages[0], self.pageidx)
        self.close_chapitre()

    def close_chapitre(self):
        """Clore le dernier chapitre (et section, etc)"""
        if self.chapitre:
            self.chapitre.pages = (self.chapitre.pages[0], self.pageidx)
            self.chapitre.articles = (self.chapitre.articles[0], len(self.articles))
        self.chapitre = None
        self.close_section()

    def close_section(self):
        """Clore la derniere section (et sous-section, etc)"""
        if self.section:
            self.section.pages = (self.section.pages[0], self.pageidx)
            self.section.articles = (self.section.articles[0], len(self.articles))
        self.section = None
        self.close_sous_section()

    def close_sous_section(self):
        """Clore la derniere sous-section"""
        if self.sous_section:
            self.sous_section.pages = (self.sous_section.pages[0], self.pageidx)
            self.sous_section.articles = (
                self.sous_section.articles[0],
                len(self.articles),
            )
        self.sous_section = None

    def close_article(self):
        """Clore le dernier article"""
        if self.articles:
            self.articles[-1].pages = (self.articles[-1].pages[0], self.pageidx)

    def extract_chapitre(self) -> Optional[models.Chapitre]:
        texte = self.pages[self.pageidx]
        m = re.search(r"CHAPITRE\s+(\d+)\s+([^\.\d]+)(\d+)?$", texte)
        if m is None:
            return None
        numero = m.group(1)
        chapitre = models.Chapitre(
            numero=numero, titre=m.group(2), pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_chapitre()
        self.chapitre = chapitre
        self.chapitres.append(self.chapitre)
        return chapitre

    def extract_annexe(self) -> Optional[models.Annexe]:
        texte = self.pages[self.pageidx]
        m = re.search(r"ANNEXE\s+(\S+)(?: –)?\s+([^\.\d]+)(\d+)?$", texte)
        if m is None:
            return None
        numero = m.group(1)
        annexe = models.Annexe(
            numero=numero, titre=m.group(2), pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        # Mettre à jour les indices de pages de la derniere annexe, chapitre, section, etc
        self.close_annexe()
        self.annexe = annexe
        self.annexes.append(self.annexe)
        return annexe

    def extract_section(self, ligne: str) -> Optional[models.Section]:
        m = re.match(r"SECTION\s+(\d+)\s+(.*)", ligne)
        if m is None:
            return None
        sec = m.group(1)
        section = models.Section(
            numero=sec,
            titre=m.group(2),
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_section()
        self.chapitre.sections.append(section)
        self.section = section
        return section

    def extract_sous_section(self, ligne: str) -> Optional[models.SousSection]:
        m = re.match(r"SOUS-SECTION\s+\d+\.(\d+)\s+(.*)", ligne)
        if m is None:
            return None
        sec = m.group(1)
        sous_section = models.SousSection(
            numero=sec,
            titre=m.group(2),
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_sous_section()
        self.section.sous_sections.append(sous_section)
        self.sous_section = sous_section
        return sous_section

    def extract_article(self, ligne: str) -> Optional[models.Article]:
        m = re.match(r"(\d+)\.\s+(.*)", ligne)
        if m is None:
            return None
        num = int(m.group(1))
        if num <= self.artidx:
            # C'est plutôt une énumération quelconque, traiter comme un alinéa
            return None
        self.close_article()
        self.artidx = num
        article = models.Article(
            chapitre=len(self.chapitres) - 1,
            section=(len(self.chapitre.sections) - 1 if self.chapitre else -1),
            sous_section=(len(self.section.sous_sections) - 1 if self.section else -1),
            numero=num,
            titre=m.group(2),
            pages=(self.pageidx, -1),
            alineas=[],
        )
        self.articles.append(article)
        return article

    def extract_text(self, pages: Optional[List[str]] = None) -> models.Reglement:
        """Extraire la structure d'un règlement d'urbanisme du texte des pages."""
        if pages is not None:
            self.pages = pages
        self.numero, self.objet = extract_title(self.pages)
        self.dates = extract_dates(self.pages)

        # Passer les tables de contenus pour trouver le premier chapitre
        self.pageidx = 0
        while self.pageidx < len(self.pages):
            if self.extract_chapitre():
                # Passer à la prochaine page
                self.pageidx += 1
                break
            self.pageidx += 1

        # Continuer pour extraire les contenus
        while self.pageidx < len(self.pages):
            if self.extract_chapitre():
                # Passer à la prochaine page
                self.pageidx += 1
            if self.extract_annexe():
                # Passer à la prochaine page
                self.pageidx += 1

            # Il devrait y en avoir rendu ici
            assert self.chapitres
            assert self.chapitre is not None

            texte = self.pages[self.pageidx]
            for line in texte.split("\n"):
                # Les annexes se trouvent à la fin du document, elles
                # vont simplement ramasser tout le contenu...
                if self.annexe:
                    self.annexe.alineas.append(line)
                    continue
                # Détecter sections et sous-sections
                if self.extract_section(line):
                    continue
                if self.extract_sous_section(line):
                    continue
                # Trouver des articles
                if self.extract_article(line):
                    continue
                # Ajouter du contenu (à un article, on espère)
                if self.articles:
                    self.articles[-1].alineas.append(line)
            self.pageidx += 1
        # Clore toutes les annexes, sections, chapitres, etc
        self.close_annexe()
        self.close_article()
        return models.Reglement(
            fichier=self.fichier,
            numero=self.numero,
            objet=self.objet,
            dates=self.dates,
            chapitres=self.chapitres,
            articles=self.articles,
            annexes=self.annexes,
        )

    def __call__(self, pdf: Path, fichier: Optional[str] = None) -> models.Reglement:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        self.fichier = pdf.name if fichier is None else fichier
        self.pages = extract_text_from_pdf(pdf)
        return self.extract_text()


def main():
    parser = make_argparse()
    args = parser.parse_args()
    ex = Extracteur()
    if args.text:
        print(json.dumps(extract_text_from_pdf(args.pdf), indent=2, ensure_ascii=False))
    else:
        print(ex(args.pdf).json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
