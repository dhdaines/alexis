from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, SerializeAsAny


class Ancrage(BaseModel):
    """Représente une partie du texte d'un document, soit un chapitre ou une section."""

    numero: str = Field(
        description="Numéro de ce chapitre ou section tel qu'il apparaît dans le texte"
    )
    titre: str
    pages: Tuple[int, int] = Field(
        description="Première et dernière indices de pages (en partant de 0) de cette partie"
    )
    textes: Tuple[int, int] = Field(
        (-1, -1),
        description="Première et dernière indices de textes",
    )


class SousSection(Ancrage):
    """Sous-section du texte.

    Le numéro comprend aussi celui de la section, par exemple
    'SOUS-SECTION 3.1 GROUPE HABITATION'
    """

    pass


class Section(Ancrage):
    """Section du texte.

    Le numéro ne comprend pas celui du chapitre, par exemple 'SECTION
    3 CLASSIFICATION DES USAGES'
    """

    sous_sections: List[SousSection] = []


class Chapitre(Ancrage):
    """Chapitre du texte."""

    sections: List[Section] = []


class Contenu(BaseModel):
    """Modèle de base pour un élément du contenu, dont un alinéa, une énumération,
    un tableau, une image, etc."""

    texte: str = Field(description="Texte indexable pour ce contenu")


class Tableau(Contenu):
    """Tableau, représenté pour le moment en image (peut-être HTML à l'avenir)"""

    tableau: Path = Field(description="Fichier avec la représentation du tableau")


class Texte(BaseModel):
    """Modèle de base pour un unité atomique (indexable) de texte, dont un
    article, une liste d'attendus, ou un annexe.
    """

    titre: Optional[str] = None
    pages: Tuple[int, int]
    contenu: List[SerializeAsAny[Contenu]] = Field(
        [], description="Contenus (alinéas, tableaux, images) de ce texte"
    )


class Annexe(Texte):
    """Annexe d'un document ou règlement."""

    annexe: str = Field(description="Numéro de cet annexe")


class Attendus(Texte):
    """Attendus d'un reglement ou resolution."""

    attendu: bool = True


class Article(Texte):
    """Article du texte."""

    article: int = Field(
        description="Numéro de cet article tel qu'il apparaît dans le texte, ou -1 pour un article sans numéro"
    )
    sous_section: int = Field(
        -1,
        description="Indice (en partant de 0) de la sous-section dans laquelle cet article apparaît, ou -1 s'il n'y en a pas",
    )
    section: int = Field(
        -1,
        description="Indice (en partant de 0) de la section dans laquelle cet article apparaît, ou -1 s'il n'y en a pas",
    )
    chapitre: int = Field(
        -1,
        description="Indice (en partant de 0) du chapitre dans laquelle cet article apparaît, ou -1 s'il n'y en a pas",
    )


class Dates(BaseModel):
    """Dates de publication ou adoption d'un document ou règlement."""

    adoption: str = Field(
        description="Date de l'adoption d'un règlement ou résolution, ou de publication d'autre document"
    )
    projet: Optional[str] = Field(
        None, description="Date de l'adoption d'un projet de règlement"
    )
    avis: Optional[str] = Field(
        None, description="Date de l'avis de motion pour un règlement"
    )
    entree: Optional[str] = Field(
        None, description="Date d'entrée en vigueur d'un règlement"
    )
    publique: Optional[str] = Field(
        None, description="Date de la consultation publique"
    )
    ecrite: Optional[str] = Field(None, description="Date de la consultation écrite")
    mrc: Optional[str] = Field(
        None, description="Date du certificat de conformité de la MRC"
    )


class Document(BaseModel):
    """Document municipal générique."""

    fichier: Path = Field(description="Nom du fichier source PDF du document")
    titre: Optional[str] = Field(
        None, description="Titre du document (tel qu'il apparaît sur le site web)"
    )
    chapitres: List[Chapitre] = []
    textes: List[Texte] = []


class Reglement(Document):
    """Structure et contenu d'un règlement."""

    numero: str = Field(description="Numéro du règlement, e.g. 1314-Z-09")
    objet: Optional[str] = Field(
        None, description="Objet du règlement, e.g. 'Lotissement'"
    )
    dates: Dates
