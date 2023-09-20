"""
Analyser un document étiquetté pour en extraire la structure.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, NamedTuple

T_obj = dict[str, Any]


@dataclass
class Bloc:
    name: str
    contents: list[T_obj]

    @property
    def texte(self) -> str:
        return " ".join(x["text"] for x in self.contents)

    @property
    def xml(self) -> str:
        return f"<{self.name}>{self.texte}</{self.name}>"


def group_iob(words: Iterable[T_obj]) -> Iterator[Bloc]:
    """Regrouper mots en blocs de texte selon leurs étiquettes IOB."""
    bloc = Bloc("", [])
    for word in words:
        bio, sep, tag = word["tag"].partition("-")
        if bio in ("B", "O"):
            if bloc.name != "":
                yield bloc
            # Could create an empty tag if this is O
            bloc = Bloc(name=tag, contents=[])
        elif bio == "I":
            # Sometimes we are missing an initial B
            if bloc.name == "":
                bloc.name = tag
        else:
            raise ValueError("Tag %s n'est pas I, O ou B" % word["tag"])
        if bio != "O":
            bloc.contents.append(word)
    if bloc.name != "":
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
        if bloc.name in PALIERS:
            element = Element(
                palier=bloc.name,
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

    def element_xml(self, el: Element, indent: int = 0) -> list[str]:
        """Représentation structurel d'un élément."""
        spacing = " " * indent
        lines = [spacing + f"<{el.palier} titre='{el.titre}'>"]
        idx = el.debut
        fin = len(self.blocs) if el.fin == -1 else el.fin
        subidx = 0
        sub = el.sub[subidx] if subidx < len(el.sub) else None
        while idx < fin:
            if sub is not None and idx == sub.debut:
                lines.extend(self.element_xml(sub, indent + 2))
                idx = len(self.blocs) if sub.fin == -1 else sub.fin
                subidx += 1
                sub = el.sub[subidx] if subidx < len(el.sub) else None
            else:
                subspacing = " " * (indent + 2)
                lines.append(subspacing + self.blocs[idx].xml)
                idx += 1
        lines.append(spacing + f"</{el.palier}>")
        return lines

    @property
    def xml(self) -> str:
        """Représentation structurel du document."""
        return "\n".join(self.element_xml(self.paliers["Document"][0]))


class Analyseur:
    """Analyse d'un document étiqueté en IOB."""

    def __call__(self, words: Iterable[T_obj]) -> Document:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        doc = Document()
        for bloc in group_iob(words):
            doc.add_bloc(bloc)
        return doc
