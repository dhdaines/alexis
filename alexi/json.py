"""
Extraire la structure du document à partir de CSV étiqueté
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from alexi.label import line_breaks
from alexi.types import (Annexe, Article, Attendus, Chapitre, Contenu, Dates,
                         Reglement, Section, SousSection, Tableau, Texte)

LOGGER = logging.getLogger("json")


class Formatteur:
    fichier: Path
    numero: str = "INCONNU"
    objet: Optional[str] = None
    titre: Optional[str] = None
    chapitre: Optional[Chapitre] = None
    section: Optional[Section] = None
    sous_section: Optional[SousSection] = None
    annexe: Optional[Annexe] = None
    article: Optional[Article] = None
    attendus: Optional[Attendus] = None
    artidx: int = -1
    pageidx: int = -1
    tableidx: int = -1
    sous_section_idx: int = 0
    chapitre_idx: int = 1

    def __init__(self, fichier: Path):
        self.fichier = Path(fichier)
        self.pages: List[str] = []
        self.chapitres: List[Chapitre] = []
        self.textes: List[Texte] = []
        self.dates: Dict[str, str] = {"Adoption": "INCONNU"}

    def process_bloc(self, tag: str, bloc: List[dict]):
        """Ajouter un bloc de texte au bon endroit."""
        texte = "\n".join(
            " ".join(w["text"] for w in line) for line in line_breaks(bloc)
        )
        contenu = Contenu(texte=texte)
        # Ne sera utilisé que si on n'a pas de structure défini
        paragraphe = Texte(pages=(self.pageidx, self.pageidx), contenu=[contenu])
        if tag == "Titre":
            self.extract_titre(texte)
        elif tag in (
            "Avis",
            "Projet",
            "Adoption",
            "Vigueur",
            "Publique",
            "Ecrite",
            "MRC",
        ):
            self.extract_date(tag, texte)
        elif tag == "Chapitre":
            self.extract_chapitre(texte)
        elif tag == "Section":
            self.extract_section(texte)
        elif tag == "SousSection":
            self.extract_sous_section(texte)
        elif tag == "Annexe":
            self.extract_annexe(texte)
        elif tag == "Article":
            if self.extract_article(texte) is None:
                self.textes.append(paragraphe)
        elif tag == "Attendu":
            self.extract_attendu(contenu)
        elif tag == "Tableau":
            assert self.tableidx > 0
            tableau = Tableau(
                texte=texte,
                tableau=Path(self.fichier.stem)
                / f"page{self.pageidx}-table{self.tableidx}.png",
            )
            LOGGER.info("Tableau sur page %d: %s", self.pageidx, tableau.tableau)
            self.tableidx += 1
            if self.annexe:
                self.annexe.contenu.append(tableau)
            elif self.article:
                self.article.contenu.append(tableau)
            else:
                paragraphe = Texte(
                    pages=(self.pageidx, self.pageidx), contenu=[tableau]
                )
                self.textes.append(paragraphe)
        elif tag in ("Alinea", "Enumeration"):
            # FIXME: à l'avenir on va prendre en charge les listes/énumérations
            if self.annexe:
                self.annexe.contenu.append(contenu)
            elif self.article:
                self.article.contenu.append(contenu)
            else:
                self.textes.append(paragraphe)

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
            self.chapitre.textes = (self.chapitre.textes[0], len(self.textes) - 1)
        self.chapitre = None
        self.close_section()

    def close_section(self):
        """Clore la derniere section (et sous-section, etc)"""
        if self.section:
            self.section.pages = (self.section.pages[0], self.pageidx)
            self.section.textes = (self.section.textes[0], len(self.textes) - 1)
        self.section = None
        self.close_sous_section()

    def close_sous_section(self):
        """Clore la derniere sous-section"""
        if self.sous_section:
            self.sous_section.pages = (self.sous_section.pages[0], self.pageidx)
            self.sous_section.textes = (
                self.sous_section.textes[0],
                len(self.textes) - 1,
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
            re.IGNORECASE | re.DOTALL,
        ):
            self.objet = m.group(1)
            self.numero = m.group(2)
            self.titre = re.sub(r"\s+", " ", m.group(0))
        elif m := re.search(
            r"règlement(?:\s+numéro)?\s+(\S+)(?:\s+(?:de|d'|sur|relatif\saux|concernant))?\s+(.*)",
            texte,
            re.IGNORECASE | re.DOTALL,
        ):
            self.titre = m.group(0)
            self.numero = m.group(1)
            self.objet = m.group(2)
        elif m := re.search(
            r"règlement(?:\s+numéro)?\s+(\S+)",
            texte,
            re.IGNORECASE,
        ):
            self.titre = m.group(0)
            self.numero = m.group(1)
        elif m := re.match(
            r"(?:de|d'|sur|relatif\saux|concernant)\s+(.*)",
            texte,
            re.IGNORECASE | re.DOTALL,
        ):
            self.titre = "\n".join((self.titre, m.group(0)))
            self.objet = m.group(1)
        else:
            self.titre = texte

    def extract_date(self, tag: str, texte: str):
        self.dates[tag] = texte

    def extract_chapitre(self, texte) -> Optional[Chapitre]:
        m = re.match(r"(?:chapitre\s+)?(\d+)\s+(.*)$", texte, re.IGNORECASE | re.DOTALL)
        if m is None:
            # texte "CHAPITRE" manquant (c'est un osti d'image, WTF!)
            # FIXME: detecter l'image en question dans convert.py
            titre = texte
            numero = str(self.chapitre_idx)
            self.chapitre_idx += 1
        else:
            numero = m.group(1)
            titre = m.group(2)
        chapitre = Chapitre(
            numero=numero,
            titre=titre,
            pages=(self.pageidx, self.pageidx),
            textes=(len(self.textes), len(self.textes)),
        )
        self.close_chapitre()
        self.chapitre = chapitre
        self.chapitres.append(self.chapitre)
        return chapitre

    def extract_annexe(self, texte) -> Optional[Annexe]:
        m = re.match(r"annexe\s+(\S+)(?: –)?\s+(.*)$", texte, re.IGNORECASE | re.DOTALL)
        if m is None:
            return None
        numero = m.group(1)
        titre = m.group(2)
        annexe = Annexe(
            annexe=numero,
            titre=titre,
            pages=(self.pageidx, self.pageidx),
        )
        # Mettre à jour les indices de pages de la derniere annexe, chapitre, section, etc
        self.close_annexe()
        self.annexe = annexe
        self.textes.append(self.annexe)
        return annexe

    def extract_section(self, ligne: str) -> Optional[Section]:
        m = re.match(r"(?:section\s+)?(\d+)\s+(.*)", ligne, re.IGNORECASE | re.DOTALL)
        if m is None:
            return None
        sec = m.group(1)
        titre = m.group(2)
        section = Section(
            numero=sec,
            titre=titre,
            pages=(self.pageidx, self.pageidx),
            textes=(len(self.textes), len(self.textes)),
        )
        self.close_section()
        if self.chapitre:
            self.chapitre.sections.append(section)
        self.section = section
        self.sous_section_idx = 1
        return section

    def extract_sous_section(self, ligne: str) -> Optional[SousSection]:
        m = re.match(
            r"(?:sous-section\s+)\d+\.(\d+)\s+(.*)", ligne, re.IGNORECASE | re.DOTALL
        )
        if m is None:
            # texte "SOUS-SECTION" manquant (c'est un osti d'image, WTF!)
            # FIXME: detecter l'image en question dans convert.py
            titre = ligne
            sec = str(self.sous_section_idx)
            self.sous_section_idx += 1
        else:
            sec = m.group(1)
            titre = m.group(2)
        sous_section = SousSection(
            numero=sec,
            titre=titre,
            pages=(self.pageidx, self.pageidx),
            textes=(len(self.textes), len(self.textes)),
        )
        self.close_sous_section()
        if self.section:
            self.section.sous_sections.append(sous_section)
        self.sous_section = sous_section
        return sous_section

    def new_article(self, num: int, titre: str) -> Article:
        article = Article(
            chapitre=len(self.chapitres) - 1,
            section=(len(self.chapitre.sections) - 1 if self.chapitre else -1),
            sous_section=(len(self.section.sous_sections) - 1 if self.section else -1),
            article=num,
            titre=titre,
            pages=(self.pageidx, self.pageidx),
            alineas=[],
        )
        self.close_article()
        self.textes.append(article)
        self.article = article
        return article

    def extract_article(self, ligne: str) -> Optional[Article]:
        m = re.match(r"(?:article\s+)?(\d+)\s*[\.:]?\s*(.*)", ligne, re.IGNORECASE)
        if m is None:
            return None
        num = int(m.group(1))
        if num <= self.artidx:
            # C'est une énumération mal étiquetée
            return None
        self.artidx = num
        return self.new_article(num, m.group(2))

    def extract_attendu(self, contenu: Contenu):
        if self.attendus is None:
            self.attendus = Attendus(
                attendu=True, pages=(self.pageidx, self.pageidx), contenu=[contenu]
            )
        else:
            self.attendus.contenu.append(contenu)
        return self.attendus

    def make_dates(self) -> Dates:
        return Dates(
            adoption=self.dates["Adoption"],
            projet=self.dates.get("Projet"),
            avis=self.dates.get("Avis"),
            entree=self.dates.get("Vigueur"),
            publique=self.dates.get("Publique"),
            ecrite=self.dates.get("Ecrite"),
            mrc=self.dates.get("MRC"),
        )

    def extract_text(self, rows: Optional[List[dict]] = None) -> Reglement:
        """Convertir une sequence de mots d'un CSV en structure"""
        if rows is not None:
            self.rows = rows
        bloc: List[dict] = []
        tag = newtag = "O"
        # Extraire les blocs de texte etiquettes
        for word in self.rows:
            label = word["tag"]
            page = int(word["page"])
            if page != self.pageidx:
                self.pageidx = page
                self.tableidx = 1
            if label == "":  # Should not happen! Ignore!
                continue
            if label != "O":
                newtag = label.partition("-")[2]
            if label[0] in ("B", "O") or newtag != tag:
                if bloc and tag:
                    self.process_bloc(tag, bloc)
                    bloc = []
                tag = newtag
            if label != "O":
                bloc.append(word)
        if bloc:
            self.process_bloc(tag, bloc)
        # Clore toutes les annexes, sections, chapitres, etc
        self.close_annexe()
        self.close_article()
        return Reglement(
            fichier=self.fichier,
            titre=self.titre,
            numero=self.numero,
            objet=self.objet,
            dates=self.make_dates(),
            chapitres=self.chapitres,
            attendus=self.attendus,
            textes=self.textes,
        )

    def __call__(self, words: Iterable[dict[str, Any]]) -> Reglement:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        self.rows = list(words)
        return self.extract_text()
