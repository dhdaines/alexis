"""
Analyser un document étiquetté pour en extraire la structure.
"""

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

from .convert import bbox_contains
from .types import Bloc, T_obj, T_bbox

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


def bbox_between(bbox: T_bbox, a: T_bbox, b: T_bbox) -> bool:
    """Déterminer si une BBox se trouve entre deux autres."""
    _, top, _, bottom = bbox
    return top >= a[1] and bottom <= b[3]


class Analyseur:
    """Analyse d'un document étiqueté en IOB."""

    def __call__(
        self,
        words: Iterable[T_obj],
        tables: Iterable[Bloc] = (),
        figures: Iterable[Bloc] = (),
    ) -> Document:
        """Extraire la structure d'un règlement d'urbanisme d'un PDF."""
        doc = Document()
        # Group tables and figures by page
        tf_blocs: defaultdict[int, list[Bloc]] = defaultdict(list)
        for bloc in itertools.chain(tables, figures):
            tf_blocs[bloc.page_number].append(bloc)
        # Store all inputs as we will do two passes
        word_sequence = list(words)
        # Get metadata
        for bloc in group_iob(word_sequence, "sequence"):
            if bloc.type not in doc.meta:
                LOGGER.info(f"{bloc.type}: {bloc.texte}")
                doc.meta[bloc.type] = bloc.texte
        # Group block-level text elements by page
        for page, blocs in itertools.groupby(
            group_iob(word_sequence), lambda x: int(x.page_number)
        ):
            seen_tf_blox: set[Bloc] = set()
            blox = list(blocs)
            # No text, add all tables/images
            if not blox:
                for tf_bloc in tf_blocs[page]:
                    doc.add_bloc(tf_bloc)
                continue
            # Consume any tables/figures before text
            for tf_bloc in tf_blocs[page]:
                _, _, _, bottom = tf_bloc.bbox
                _, top, _, _ = blox[0].bbox
                if bottom <= top:
                    doc.add_bloc(tf_bloc)
                    seen_tf_blox.add(tf_bloc)
            # Add text/table/figure blocs
            prev_bloc: Optional[Bloc] = None
            for bloc in blox:
                found_bloc = False
                for tf_bloc in tf_blocs[page]:
                    if bbox_contains(tf_bloc.bbox, bloc.bbox):
                        found_bloc = True
                        tf_bloc.contenu.extend(bloc.contenu)
                        if tf_bloc not in seen_tf_blox:
                            LOGGER.info(
                                "Page %d contain tf bloc %s %s",
                                page,
                                tf_bloc.type,
                                tf_bloc.bbox,
                            )
                            doc.add_bloc(tf_bloc)
                            seen_tf_blox.add(tf_bloc)
                        break
                    elif prev_bloc and bbox_between(
                        tf_bloc.bbox, prev_bloc.bbox, bloc.bbox
                    ):
                        if tf_bloc not in seen_tf_blox:
                            LOGGER.info(
                                "Page %d between tf bloc %s %s",
                                page,
                                tf_bloc.type,
                                tf_bloc.bbox,
                            )
                            doc.add_bloc(tf_bloc)
                            seen_tf_blox.add(tf_bloc)
                if not found_bloc:
                    LOGGER.info("Page %d bloc %s %s", page, bloc.type, bloc.bbox)
                    doc.add_bloc(bloc)
                prev_bloc = bloc
            # Add any tables or figures that might remain at the bottom of the page
            for tf_bloc in tf_blocs[page]:
                if tf_bloc not in seen_tf_blox:
                    LOGGER.info(
                        "Page %d extra tf bloc %s %s", page, tf_bloc.type, tf_bloc.bbox
                    )
                    doc.add_bloc(tf_bloc)
        return doc
