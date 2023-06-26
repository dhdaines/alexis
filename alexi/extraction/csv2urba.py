#!/usr/bin/env python3

"""
Convertir CSV du traitement automatique en JSON
"""

import argparse
import re
from csv import DictReader
from pathlib import Path
from typing import Optional, List

from alexi import models


def make_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Fichier CSV", type=Path)
    return parser


class Converteur:
    fichier: str = "INCONNU"
    numero: str = "INCONNU"
    objet: str = "INCONNU"
    titre: str = "INCONNU"
    dates: models.Dates = models.Dates(adoption="INCONNU")
    chapitre: Optional[models.Chapitre] = None
    section: Optional[models.Section] = None
    sous_section: Optional[models.SousSection] = None
    annexe: Optional[models.Annexe] = None
    article: Optional[models.Article] = None
    artidx: int = -1
    pageidx: int = -1

    def __init__(self, fichier: str = None):
        if fichier is not None:
            self.fichier = fichier
        self.pages: List[str] = []
        self.chapitres: List[models.Chapitre] = []
        self.articles: List[models.Article] = []
        self.annexes: List[models.Annexe] = []
        self.dates

    def process_bloc(self, tag: str, bloc: str):
        """Ajouter un bloc de texte au bon endroit."""
        if tag == "Titre" and self.titre == "INCONNU":
            self.titre = bloc
        elif tag == "Chapitre":
            self.extract_chapitre(bloc)
        elif tag == "Section":
            self.extract_section(bloc)
        elif tag == "SousSection":
            self.extract_sous_section(bloc)
        elif tag == "Annexe":
            self.extract_annexe(bloc)
        elif tag == "Article":
            self.extract_article(bloc)
        elif tag in ("Alinea", "Enumeration"):
            if self.article:
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

    def extract_chapitre(self, texte) -> Optional[models.Chapitre]:
        m = re.match(r"CHAPITRE\s+(\d+)\s+(.*)$", texte)
        if m is None:
            return None
        numero = m.group(1)
        titre = m.group(2)
        chapitre = models.Chapitre(
            numero=numero, titre=titre, pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_chapitre()
        self.chapitre = chapitre
        self.chapitres.append(self.chapitre)
        return chapitre

    def extract_annexe(self, texte) -> Optional[models.Annexe]:
        m = re.search(r"ANNEXE\s+(\S+)(?: –)?\s+([^\.\d]+)(\d+)?$", texte)
        if m is None:
            return None
        numero = m.group(1)
        titre = re.sub(r"\s+", " ", m.group(2).strip())
        annexe = models.Annexe(
            numero=numero, titre=titre, pages=(self.pageidx, -1),
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
        titre = re.sub(r"\s+", " ", m.group(2).strip())
        section = models.Section(
            numero=sec,
            titre=titre,
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
        titre = re.sub(r"\s+", " ", m.group(2).strip())
        sous_section = models.SousSection(
            numero=sec,
            titre=titre,
            pages=(self.pageidx, -1),
            articles=(len(self.articles), -1),
        )
        self.close_sous_section()
        self.section.sous_sections.append(sous_section)
        self.sous_section = sous_section
        return sous_section

    def new_article(self, num: int, titre: str) -> models.Article:
        article = models.Article(
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

    def extract_article(self, ligne: str) -> Optional[models.Article]:
        m = re.match(r"(\d+)\.\s+(.*)", ligne)
        if m is None:
            # On le reconnaît pas comme un article, c'est peut-être un
            # sous-titre, on regardera ça plus tard
            return None
        num = int(m.group(1))
        if num <= self.artidx:
            # C'est plutôt une énumération quelconque, traiter comme un alinéa
            return None
        self.artidx = num
        return self.new_article(num, m.group(2))

    def extract_text(self, rows: Optional[List[dict]] = None) -> models.Reglement:
        """Convertir une sequence de mots d'un CSV en structure"""
        if rows is not None:
            self.rows = rows
        bloc = []
        tag = None
        # Extraire les blocs de texte etiquettes
        for row in self.rows:
            label = row["tag"]
            word = row["text"]
            if label[0] == 'B':
                if bloc:
                    self.process_bloc(tag, " ".join(bloc))
                    bloc = []
                tag = label.partition('-')[2]
            elif len(bloc) == 0:
                tag = label.partition('-')[2]
            bloc.append(word)
        self.process_bloc(tag, " ".join(bloc))
        # Clore toutes les annexes, sections, chapitres, etc
        self.close_annexe()
        self.close_article()
        return models.Reglement(
            fichier=self.fichier,
            titre=self.titre,
            numero=self.numero,
            objet=self.objet,
            dates=self.dates,
            chapitres=self.chapitres,
            articles=self.articles,
            annexes=self.annexes,
        )

    def __call__(self, csv: Path, fichier: Optional[str] = None) -> models.Reglement:
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
