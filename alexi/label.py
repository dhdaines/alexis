"""Classification des unités de texte en format CSV"""

import csv
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Iterator, TextIO

LOGGER = logging.getLogger("label")


def group_paragraphs(
    words: Iterable[dict[str, Any]]
) -> Iterator[tuple[str, Iterable[dict[str, Any]]]]:
    """Grouper le flux de mots par unité (BI*) de texte."""
    bio = tag = "O"
    paragraph = []
    for word in words:
        next_bio, _, next_tag = word["tag"].partition("-")
        if not next_tag:
            next_tag = next_bio
        if next_bio != bio or next_bio == "B":
            if next_bio == "I":
                paragraph.append(word)
            else:
                # Could do this without storing the paragraph, but not
                # strictly necessary (it's not that big!)
                if paragraph:
                    yield tag, paragraph
                paragraph = [word]
                # Note: tag *could* change inside a block
                tag = next_tag
            bio = next_bio
        else:
            paragraph.append(word)
    if paragraph:
        yield tag, paragraph


class Classificateur:
    def classify_paragraph_heuristic(
        self, tag: str, paragraph: Sequence[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Classification heuristique très simplistique d'un alinéa, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption.
        """
        if len(paragraph) == 0:
            return
        word = paragraph[0]
        if word["text"].lower() == "article":
            tag = "Article"
        elif word["text"].lower() == "attendu":
            tag = "Attendu"
            for idx in range(len(paragraph) - 3):
                if [
                    w["text"].lower() for w in paragraph[idx : idx + 3]
                ] == "avis de motion".split():
                    tag = "Avis"
        if tag == "O":
            word["tag"] = "O"
        else:
            word["tag"] = f"B-{tag}"
        yield word
        for word in paragraph[1:]:
            if tag == "O":
                word["tag"] = "O"
            else:
                word["tag"] = f"I-{tag}"
            yield word

    def classify_heuristic(
        self, words: Iterable[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """Classification heuristique très simplistique des alinéas, suffisant
        pour détecter les articles, énumérations, et parfois les dates
        d'adoption."""
        for tag, paragraph in group_paragraphs(words):
            if not paragraph:
                return
            for word in self.classify_paragraph_heuristic(tag, paragraph):
                yield word

    def __call__(self, infh: TextIO) -> list[dict[str, Any]]:
        reader = csv.DictReader(infh)
        words = self.classify_heuristic(reader)
        return list(words)
