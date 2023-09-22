"""
Analyser un document étiquetté pour en extraire la structure.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from pdfplumber.utils.geometry import T_bbox, merge_bboxes

T_obj = dict[str, Any]


@dataclass
class Bloc:
    """Élément de présentation (bloc de texte ou image)"""

    type: str
    contenus: list[T_obj]

    @property
    def texte(self) -> str:
        """Représentation textuel du bloc."""
        return " ".join(x["text"] for x in self.contenus)

    @property
    def bbox(self) -> T_bbox:
        return merge_bboxes(
            (int(word["x0"]), int(word["top"]), int(word["x1"]), int(word["bottom"]))
            for word in self.contenus
        )


def group_iob(words: Iterable[T_obj]) -> Iterator[Bloc]:
    """Regrouper mots en blocs de texte selon leurs étiquettes IOB."""
    bloc = Bloc(type="", contenus=[])
    for word in words:
        bio, sep, tag = word["tag"].partition("-")
        if bio in ("B", "O"):
            if bloc.type != "":
                yield bloc
            # Could create an empty tag if this is O
            bloc = Bloc(type=tag, contenus=[])
        elif bio == "I":
            # Sometimes we are missing an initial B
            if bloc.type == "":
                bloc.type = tag
        else:
            raise ValueError("Tag %s n'est pas I, O ou B" % word["tag"])
        if bio != "O":
            bloc.contenus.append(word)
    if bloc.type != "":
        yield bloc


PALIERS = [
    "Document",
    "Chapitre",
    "Section",
    "SousSection",
    "Article",
]


@dataclass
class Element:
    palier: str
    titre: str
    debut: int
    fin: int
    sub: list["Element"]


class Document:
    """Document avec blocs de texte et structure."""

    def __init__(self) -> None:
        self.blocs: list[Bloc] = []
        self.paliers: defaultdict[str, list[Element]] = defaultdict(list)
        doc = Element(palier="Document", titre="", debut=0, fin=-1, sub=[])
        self.paliers["Document"].append(doc)

    def add_bloc(self, bloc: Bloc):
        """Ajouter un bloc de texte."""
        if bloc.type in PALIERS:
            element = Element(
                palier=bloc.type,
                titre=bloc.texte,
                debut=len(self.blocs),
                fin=-1,
                sub=[],
            )
            self.add_element(element)
        else:
            self.blocs.append(bloc)

    def add_element(self, element: Element):
        """Ajouter un élément au palier approprié."""
        # Fermer l'élément précédent du paliers actuel et inférieurs
        pidx = PALIERS.index(element.palier)
        for palier in PALIERS[pidx:]:
            if self.paliers[palier]:
                previous = self.paliers[palier][-1]
                if previous.fin == -1:
                    previous.fin = element.debut
        # Ajouter l'élément au palier actuel
        self.paliers[element.palier].append(element)
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
    def root(self) -> Element:
        """Racine de l'arborescence structurel du document."""
        return self.paliers["Document"][0]


def bbox_between(bbox: T_bbox, a: T_bbox, b: T_bbox) -> bool:
    """Déterminer si une BBox se trouve entre deux autres."""
    _, top, _, bottom = bbox
    return top >= a[1] and bottom <= b[3]


class Analyseur:
    """Analyse d'un document étiqueté en IOB."""

    def __call__(self, words: Iterable[T_obj]) -> Document:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        doc = Document()
        for bloc in group_iob(words):
            doc.add_bloc(bloc)
        return doc
