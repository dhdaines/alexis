"""
Analyser un document étiquetté pour en extraire la structure.
"""

import itertools
import logging
import operator
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .convert import merge_overlaps
from .types import Bloc, T_obj

LOGGER = logging.getLogger("analyse")


def group_iob(words: Iterable[T_obj], key: str = "segment") -> Iterator[Bloc]:
    """Regrouper mots en blocs de texte selon leurs étiquettes IOB."""
    bloc = Bloc(type="", contenu=[])
    for word in words:
        bio, sep, tag = word[key].partition("-")
        if bio in ("B", "O"):
            if bloc.type != "":
                yield bloc
            # Could create an empty tag if this is O
            bloc = Bloc(type=tag, contenu=[])
        elif bio == "I":
            # Sometimes we are missing an initial B
            if bloc.type == "":
                bloc.type = tag
        else:
            raise ValueError("Tag %s n'est pas I, O ou B" % word[key])
        if bio != "O":
            bloc.contenu.append(word)
    if bloc.type != "":
        yield bloc


PALIERS = [
    "Document",
    "Annexe",
    "Chapitre",
    "Section",
    "SousSection",
    "Article",
]


@dataclass
class Element:
    type: str
    titre: str = ""
    numero: str = ""
    debut: int = 0
    fin: int = -1
    sub: list["Element"] = field(default_factory=list)
    page: int = 1


ELTYPE = r"(?i:article|chapitre|section|sous-section|titre|annexe)"
DOTSPACEDASH = r"(?:\.|\s*[:—–-]| )"
NUM = r"(\d+)"
NUMDOT = r"((?:\d+\.)+\d+)"
ALPHA = r"[A-Z]"
ROMAN = r"[XIV]+"
NUMRE = re.compile(
    rf"{ELTYPE}?\s*"
    r"(?:"
    rf"{NUMDOT}{DOTSPACEDASH}?"
    r"|"
    rf"{NUM}{DOTSPACEDASH}?"
    r"|"
    rf"({ALPHA}|{ROMAN}){DOTSPACEDASH}"
    r")"
    r"\s*"
)
NUMENDRE = re.compile(rf".*\b{NUM}{DOTSPACEDASH}\s*$")


class Document:
    """Document avec blocs de texte et structure."""

    unknown_id: int = 0
    fileid: str
    meta: dict[str, str]
    paliers: dict[str, list[Element]]
    contenu: list[Bloc]

    def __init__(self, fileid: str, numero: str = "", titre: str = "Document") -> None:
        self.fileid = fileid
        self.paliers = {}
        self.meta = {}
        self.contenu = []
        doc = Element(type="Document", titre=titre, numero=numero)
        self.paliers.setdefault("Document", []).append(doc)

    def extract_numero(self, titre: str) -> tuple[str, str]:
        """Extraire le numero d'un article/chapitre/section/annexe, si possible."""
        # FIXME: UNIT TEST THIS!!!
        if m := NUMRE.match(titre):
            if m.group(1):  # sous section (x.y.z)
                numero = m.group(1)
            elif m.group(2):  # article (x)
                numero = m.group(2)
            elif m.group(3):  # annexe (A), chapitre (III)
                numero = m.group(3)
            else:
                numero = "_%d" % self.unknown_id
                self.unknown_id += 1
            titre = titre[m.end(0) :]
        elif m := NUMENDRE.match(titre):
            numero = m.group(1)
            titre = titre[: m.start(1)]
        else:
            numero = "_%d" % self.unknown_id
            self.unknown_id += 1
        return numero, titre

    def add_bloc(self, bloc: Bloc):
        """Ajouter un bloc de texte."""
        if bloc.type in PALIERS:
            numero, titre = self.extract_numero(bloc.texte)
            element = Element(
                type=bloc.type,
                titre=titre,
                numero=numero,
                debut=len(self.contenu),
                fin=-1,
                sub=[],
                page=int(bloc.page_number),
            )
            self.add_element(element)
        else:
            self.contenu.append(bloc)

    def add_element(self, element: Element):
        """Ajouter un élément au palier approprié."""
        # Fermer l'élément précédent du paliers actuel et inférieurs
        pidx = PALIERS.index(element.type)
        for palier in PALIERS[pidx:]:
            if palier in self.paliers and self.paliers[palier]:
                previous = self.paliers[palier][-1]
                if previous.fin == -1:
                    previous.fin = element.debut
        # Ajouter l'élément au palier actuel
        self.paliers.setdefault(element.type, []).append(element)
        # Ajouter à un élément supérieur s'il existe est s'il est ouvert
        if pidx == 0:
            return
        for palier in PALIERS[pidx - 1 :: -1]:
            if palier in self.paliers and self.paliers[palier]:
                previous = self.paliers[palier][-1]
                if previous.fin == -1:
                    previous.sub.append(element)
                    break

    @property
    def structure(self) -> Element:
        """Racine de l'arborescence structurel du document."""
        return self.paliers["Document"][0]

    @property
    def titre(self) -> str:
        """Titre du document."""
        return self.structure.titre

    @property
    def numero(self) -> str:
        """Numero du document."""
        return self.structure.numero


class Analyseur:
    """Analyse d'un document étiqueté en IOB."""

    def __init__(self, fileid: str, words: Iterable[T_obj]):
        self.fileid = fileid
        self.words: list[T_obj] = list(words)
        self.blocs: list[Bloc] = list(group_iob(self.words, "segment"))
        self.metadata: dict[str, str] = {}
        for bloc in group_iob(self.words, "sequence"):
            if bloc.type not in self.metadata:
                LOGGER.info(f"{bloc.type}: {bloc.texte}")
                self.metadata[bloc.type] = bloc.texte

    def add_images(self, images: Iterable[Bloc], merge: bool = True):
        """Insérer les images en les fusionnant avec le texte (et entre elles)
        si demandé."""
        images_bypage: dict[int, list[Bloc]] = {
            page_number: list(group)
            for page_number, group in itertools.groupby(
                images, operator.attrgetter("page_number")
            )
        }

        # FIXME: assume that we can order things this way!
        def bbox_order(bloc):
            x0, top, x1, bottom = bloc.bbox
            return (top, x0, bottom, x1)

        new_blocs: list[Bloc] = []
        for page_number, group in itertools.groupby(
            self.blocs, operator.attrgetter("page_number")
        ):
            if page_number in images_bypage:
                page_blocs = list(group)
                page_blocs.extend(images_bypage[page_number])
                page_blocs.sort(key=bbox_order)
                if merge:
                    new_blocs.extend(merge_overlaps(page_blocs))
                else:
                    new_blocs.extend(page_blocs)
            else:
                new_blocs.extend(group)
        self.blocs = new_blocs

    def __call__(
        self,
        blocs: Optional[Iterable[Bloc]] = None,
    ) -> Document:
        """Analyse du structure d'un document."""
        titre = self.metadata.get("Titre", "Document")
        numero = self.metadata.get("Numero", "")
        if m := re.search(r"(?i:num[ée]ro)\s+([0-9][A-Z0-9-]+)", titre):
            LOGGER.info("Numéro extrait du titre: %s", m.group(1))
            numero = m.group(1)
            titre = titre[: m.start(0)] + titre[m.end(0) :]
        elif m := re.search(r"(?i:r[èe]glement)\s+([0-9][A-Z0-9-]+)", titre):
            LOGGER.info("Numéro extrait du titre: %s", m.group(1))
            numero = m.group(1)
            titre = titre[: m.start(1)] + titre[m.end(1) :]
        elif m := re.match(r".*(\b\d+-\d+-[A-Z]+$)", titre):
            LOGGER.info("Numéro extrait du titre: %s", m.group(1))
            numero = m.group(1)
            titre = titre[: m.start(1)]
        doc = Document(self.fileid, numero, titre)
        doc.meta = self.metadata
        # Group block-level text elements by page from segment tags
        if blocs is None:
            blocs = self.blocs
        for page, blocs in itertools.groupby(blocs, operator.attrgetter("page_number")):
            for bloc in blocs:
                doc.add_bloc(bloc)
        return doc
