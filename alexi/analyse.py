"""
Analyser un document étiquetté pour en extraire la structure.
"""

import itertools
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

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
    titre: str
    debut: int
    fin: int
    sub: list["Element"]


class Document:
    """Document avec blocs de texte et structure."""

    meta: dict[str, str]

    def __init__(self) -> None:
        self.contenu: list[Bloc] = []
        self.paliers: defaultdict[str, list[Element]] = defaultdict(list)
        doc = Element(type="Document", titre="", debut=0, fin=-1, sub=[])
        self.paliers["Document"].append(doc)
        self.meta = {}

    def add_bloc(self, bloc: Bloc):
        """Ajouter un bloc de texte."""
        if bloc.type in PALIERS:
            element = Element(
                type=bloc.type,
                titre=bloc.texte,
                debut=len(self.contenu),
                fin=-1,
                sub=[],
            )
            self.add_element(element)
        else:
            self.contenu.append(bloc)

    def add_element(self, element: Element):
        """Ajouter un élément au palier approprié."""
        # Fermer l'élément précédent du paliers actuel et inférieurs
        pidx = PALIERS.index(element.type)
        for palier in PALIERS[pidx:]:
            if self.paliers[palier]:
                previous = self.paliers[palier][-1]
                if previous.fin == -1:
                    previous.fin = element.debut
        # Ajouter l'élément au palier actuel
        self.paliers[element.type].append(element)
        # Ajouter à un élément supérieur s'il existe est s'il est ouvert
        if pidx == 0:
            return
        for palier in PALIERS[pidx - 1 :: -1]:
            if self.paliers[palier]:
                previous = self.paliers[palier][-1]
                if previous.fin == -1:
                    previous.sub.append(element)
                    break

    @property
    def structure(self) -> Element:
        """Racine de l'arborescence structurel du document."""
        return self.paliers["Document"][0]


class Analyseur:
    """Analyse d'un document étiqueté en IOB."""

    def __call__(
        self, words: Iterable[T_obj], blocs: Optional[Iterable[Bloc]] = None
    ) -> Document:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        doc = Document()
        # Store all inputs as we will do two passes (for sequence and segment tags)
        word_sequence = list(words)
        # Get metadata from sequence tags
        for bloc in group_iob(word_sequence, "sequence"):
            if bloc.type not in doc.meta:
                LOGGER.info(f"{bloc.type}: {bloc.texte}")
                doc.meta[bloc.type] = bloc.texte
        # Group block-level text elements by page from segment tags
        if blocs is None:
            blocs = group_iob(word_sequence)
        for page, blocs in itertools.groupby(blocs, operator.attrgetter("page_number")):
            for bloc in blocs:
                doc.add_bloc(bloc)
        return doc
