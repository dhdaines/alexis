#!/usr/bin/env python3

"""
Convertir CSV du traitement automatique en JSON
"""

import argparse
import re
from csv import DictReader
from pathlib import Path
from typing import List, Optional

from alexi import types


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Fichier CSV", type=Path)
    return parser


class Converteur:
    fichier: str = "INCONNU"
    numero: str = "INCONNU"
    objet: str = "INCONNU"
    titre: str = "INCONNU"
    adoption: str = "INCONNU"
    avis: str = "INCONNU"
    entree: str = "INCONNU"
    chapitre: Optional[types.Chapitre] = None
    section: Optional[types.Section] = None
    sous_section: Optional[types.SousSection] = None
    annexe: Optional[types.Annexe] = None
    article: Optional[types.Article] = None
    artidx: int = -1
    pageidx: int = -1

    def __init__(self, fichier: Optional[str] = None):
        if fichier is not None:
            self.fichier = fichier
        self.pages: List[str] = []
        self.chapitres: List[types.Chapitre] = []
        self.articles: List[types.Article] = []
        self.annexes: List[types.Annexe] = []

    def process_bloc(self, tag: str, bloc: str):
        """Ajouter un bloc de texte au bon endroit."""
        if tag == "Titre":
            self.extract_titre(bloc)
        elif tag == "Avis":
            self.extract_avis(bloc)
        elif tag == "Adoption":
            self.extract_adoption(bloc)
        elif tag == "Vigueur":
            self.extract_entree(bloc)
        elif tag == "Chapitre":
            self.extract_chapitre(bloc)
        elif tag == "Section":
            self.extract_section(bloc)
        elif tag == "SousSection":
            self.extract_sous_section(bloc)
        elif tag == "Annexe":
            self.extract_annexe(bloc)
        elif tag == "Article":
            if self.extract_article(bloc) is None:
                if self.article:
                    self.article.alineas.append(bloc)
        elif tag in ("Alinea", "Enumeration"):
            if self.annexe:
                self.annexe.alineas.append(bloc)
            elif self.article:
                self.article.alineas.append(bloc)

    def close_annexe(self):
        """Clore la derniere annexe (et chapitre, et section, etc)"""
        if self.annexe:
            self.annexe.pages = (self.annexe.pages[0], self.pageidx)
        self.annexe = None
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
        self.close_article()

    def close_article(self):
        """Clore le dernier article"""
        if self.article:
            self.article.pages = (self.article.pages[0], self.pageidx)
        self.article = None

    def extract_titre(self, texte: str):
        if m := re.search(
            r"règlement(?:\s+(?:de|d'|sur|relatif aux))?\s+(.*)\s+numéro\s+(\S+)",
            texte,
            re.IGNORECASE,
        ):
            self.objet = m.group(1)
            self.numero = m.group(2)
            self.titre = re.sub(r"\s+", " ", m.group(0))
        elif m := re.search(
            r"règlement\s+numéro\s+(\S+)(?:\s+(?:de|d'|sur|relatif aux))?\s+(.*)",
            texte,
            re.IGNORECASE,
        ):
            self.titre = m.group(0)
            self.numero = m.group(1)
            self.objet = m.group(2)

    def extract_avis(self, texte: str):
        self.avis = texte

    def extract_adoption(self, texte: str):
        self.adoption = texte

    def extract_entree(self, texte: str):
        self.entree = texte

    def extract_chapitre(self, texte) -> Optional[types.Chapitre]:
        m = re.match(r"(?:chapitre\s+)?(\d+)\s+(.*)$", texte, re.IGNORECASE)
        if m is None:
            return None
        numero = m.group(1)
        titre = m.group(2)
        chapitre = types.Chapitre(
            numero=numero,
            titre=titre,
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_chapitre()
        self.chapitre = chapitre
        self.chapitres.append(self.chapitre)
        return chapitre

    def extract_annexe(self, texte) -> Optional[types.Annexe]:
        m = re.match(r"annexe\s+(\S+)(?: –)?\s+(.*)$", texte, re.IGNORECASE)
        if m is None:
            return None
        numero = m.group(1)
        titre = m.group(2)
        annexe = types.Annexe(
            numero=numero,
            titre=titre,
            pages=(self.pageidx, -1),
        )
        # Mettre à jour les indices de pages de la derniere annexe, chapitre, section, etc
        self.close_annexe()
        self.annexe = annexe
        self.annexes.append(self.annexe)
        return annexe

    def extract_section(self, ligne: str) -> Optional[types.Section]:
        m = re.match(r"(?:section\s+)?(\d+)\s+(.*)", ligne, re.IGNORECASE)
        if m is None:
            return None
        sec = m.group(1)
        titre = m.group(2)
        section = types.Section(
            numero=sec,
            titre=titre,
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_section()
        if self.chapitre:
            self.chapitre.sections.append(section)
        self.section = section
        return section

    def extract_sous_section(self, ligne: str) -> Optional[types.SousSection]:
        m = re.match(r"(?:sous-section\s+)\d+\.(\d+)\s+(.*)", ligne, re.IGNORECASE)
        if m is None:
            return None
        sec = m.group(1)
        titre = m.group(2)
        sous_section = types.SousSection(
            numero=sec,
            titre=titre,
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_sous_section()
        if self.section:
            self.section.sous_sections.append(sous_section)
        self.sous_section = sous_section
        return sous_section

    def new_article(self, num: int, titre: str) -> types.Article:
        article = types.Article(
            chapitre=len(self.chapitres) - 1,
            section=(len(self.chapitre.sections) - 1 if self.chapitre else -1),
            sous_section=(len(self.section.sous_sections) - 1 if self.section else -1),
            numero=num,
            titre=titre,
            pages=(self.pageidx, -1),
            alineas=[],
        )
        self.close_article()
        self.articles.append(article)
        self.article = article
        return article

    def extract_article(self, ligne: str) -> Optional[types.Article]:
        m = re.match(r"(?:article\s+)?(\d+)\.?\s+(.*)", ligne, re.IGNORECASE)
        if m is None:
            return None
        num = int(m.group(1))
        if num <= self.artidx:
            # C'est une énumération mal étiquetée
            return None
        self.artidx = num
        return self.new_article(num, m.group(2))

    def extract_text(self, rows: Optional[List[dict]] = None) -> types.Reglement:
        """Convertir une sequence de mots d'un CSV en structure"""
        if rows is not None:
            self.rows = rows
        bloc: List[str] = []
        tag = newtag = "O"
        # Extraire les blocs de texte etiquettes
        self.pageidx = 0
        for row in self.rows:
            label = row["tag"]
            page = int(row["page"])
            word = row["text"]
            if label != "O":
                newtag = label.partition("-")[2]
            if label[0] in ("B", "O") or newtag != tag:
                if bloc and tag:
                    self.process_bloc(tag, " ".join(bloc))
                    bloc = []
                    self.pageidx = page
                tag = newtag
            if label != "O":
                bloc.append(word)
        if bloc:
            self.process_bloc(tag, " ".join(bloc))
        # Clore toutes les annexes, sections, chapitres, etc
        self.close_annexe()
        self.close_article()
        return types.Reglement(
            fichier=self.fichier,
            titre=self.titre,
            numero=self.numero,
            objet=self.objet,
            dates=types.Dates(
                adoption=self.adoption, avis=self.avis, entree=self.entree
            ),
            chapitres=self.chapitres,
            articles=self.articles,
            annexes=self.annexes,
        )

    def __call__(self, csv: Path, fichier: Optional[str] = None) -> types.Reglement:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        self.fichier = csv.name if fichier is None else fichier
        with open(csv, "rt") as infh:
            reader = DictReader(infh)
            self.rows = list(reader)
        return self.extract_text()


def main():
    parser = make_argparse()
    args = parser.parse_args()
    conv = Converteur()
    print(conv(args.csv).json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
