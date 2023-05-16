from datetime import date
from typing import Optional, List, Tuple

from pydantic import BaseModel


class Ancrage(BaseModel):
    numero: str
    page: int
    titre: Optional[str]
    debut: int = 0
    fin: int = 0

class SousSection(Ancrage):
    pass

class Section(Ancrage):
    sous_sections: List[SousSection] = []  # yes, this works

class Chapitre(Ancrage):
    sections: List[Section] = []

class Annexe(Ancrage):
    alineas: List[str] = []

class Article(BaseModel):
    numero: int
    titre: Optional[str]
    alineas: List[str] = []

class Dates(BaseModel):
    avis: Optional[str]
    adoption: str
    entree: str
    
class Reglement(BaseModel):
    numero: str
    objet: Optional[str]
    dates: Dates
    chapitres: List[Chapitre] = []
    articles: List[Article] = []
    annexes: List[Annexe] = []

