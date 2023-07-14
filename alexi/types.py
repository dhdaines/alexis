from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Ancrage(BaseModel):
    """Représente une partie du texte d'un document, soit un chapitre ou une section."""

    numero: str = Field(
        description="Numéro de ce chapitre ou section tel qu'il apparaît dans le texte"
    )
    titre: str
    pages: Tuple[int, int] = Field(
        description="Première et dernière indices de pages (en partant de 0) de cette partie"
    )
    contenus: Tuple[int, int] = Field(
        (-1, -1),
        description="Première et dernière indices de contenus (articles, alinéas, tableaux, etc)",
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
    """Modèle de base pour du contenu textuel."""

    titre: Optional[str] = None
    pages: Tuple[int, int]
    alineas: List[str] = []


class Annexe(Contenu):
    """Annexe du texte."""

    annexe: str


class Attendu(Contenu):
    """Attendu du texte."""

    pass


class Article(Contenu):
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
    contenus: List[Contenu] = []


class Reglement(Document):
    """Structure et contenu d'un règlement."""

    numero: str = Field(description="Numéro du règlement, e.g. 1314-Z-09")
    objet: Optional[str] = Field(
        None, description="Objet du règlement, e.g. 'Lotissement'"
    )
    dates: Dates
