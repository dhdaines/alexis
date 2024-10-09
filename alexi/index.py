"""
Construire un index pour faire des recherches dans les données extraites.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, Tuple

from bs4 import BeautifulSoup
from lunr import get_default_builder, lunr, trimmer  # type: ignore
from lunr.pipeline import Pipeline  # type: ignore
from unidecode import unidecode  # type: ignore

LOGGER = logging.getLogger("index")


@dataclass
class Document:
    url: str
    titre: str
    texte: str


def body_text(soup: BeautifulSoup):
    body = soup.find_all("div", id="body")
    assert body is not None
    for header in body[0](class_="header"):
        header.extract()
    for img in body[0]("img"):
        alt = soup.new_tag("p")
        alt.string = img["alt"]
        img.replace_with(alt)
    return re.sub("\n\n+", "\n\n", soup.text.strip())


def unifold(token, _idx=None, _tokens=None):
    def wrap_unidecode(text, _metadata):
        return unidecode(text)

    return token.update(wrap_unidecode)


Pipeline.register_function(unifold, "unifold")


def collect(indir: Path) -> Iterator[Tuple[str, str, str]]:
    # Use index.html to find things (as in the js version)
    LOGGER.info("Traitement de %s", indir / "index.html")
    with open(indir / "index.html", "rt") as infh:
        soup = BeautifulSoup(infh, features="lxml")
    for section in soup.select("li.node"):
        summary = section.summary
        if summary is None:
            LOGGER.error("<summary> non trouvé dans %s", section)
            continue
        title = summary.text
        if "Document" in section["class"]:
            LOGGER.info("Texte complet de %s ne sera pas indexé", title)
            continue
        a = section.a
        assert a is not None
        url = a["href"]
        assert not isinstance(url, list)
        # Assume it is a relative URL (we made it)
        LOGGER.info("Traitement de %s", indir / url)
        with open(indir / url, "rt") as infh:
            subsoup = BeautifulSoup(infh, features="lxml")
            yield url, title, body_text(subsoup)
    for text in soup.select("li.leaf"):
        assert text is not None
        a = text.a
        assert a is not None
        title = a.text
        url = a["href"]
        assert not isinstance(url, list)
        LOGGER.info("Traitement de %s", indir / url)
        with open(indir / url, "rt") as infh:
            subsoup = BeautifulSoup(infh, features="lxml")
            yield url, title, body_text(subsoup)


def index(indirs: List[Path], outdir: Path) -> None:
    """
    Generer l'index a partir des fichiers HTML.
    """
    # Metadata (use to index specific zones, etc)
    # with open(indir / "index.json", "rt") as infh:
    #     metadata = json.load(infh)

    # lunr does not do storage so we store plaintext here
    textes = []
    if len(indirs) == 1:
        # Backward compat
        for url, title, text in collect(indirs[0]):
            textes.append((url, title, text))
    else:
        for indir in indirs:
            ville = indir.name
            for url, title, text in collect(indir):
                textes.append((f"{ville}/{url}", title, text))

    outdir.mkdir(exist_ok=True)
    with open(outdir / "textes.json", "wt", encoding="utf-8") as outfh:
        json.dump(textes, outfh, indent=2, ensure_ascii=False)

    builder = get_default_builder("fr")
    # DO NOT USE the French trimmer as it is seriously defective
    builder.pipeline.remove(
        builder.pipeline.registered_functions["lunr-multi-trimmer-fr"]
    )
    builder.pipeline.before(
        builder.pipeline.registered_functions["stopWordFilter-fr"], trimmer.trimmer
    )
    # Missing pipeline functions for search
    builder.search_pipeline.before(
        builder.search_pipeline.registered_functions["stemmer-fr"],
        builder.search_pipeline.registered_functions["stopWordFilter-fr"],
    )
    builder.search_pipeline.before(
        builder.search_pipeline.registered_functions["stopWordFilter-fr"],
        trimmer.trimmer,
    )
    builder.pipeline.add(unifold)
    builder.search_pipeline.add(unifold)
    # builder.metadata_whitelist.append("position")
    LOGGER.info("pipeline: %s", builder.pipeline)
    LOGGER.info("search pipeline: %s", builder.pipeline)

    index = lunr(
        ref="idx",
        fields=[{"field_name": "titre", "boost": 2}, "texte"],
        documents=[
            {
                "idx": idx,
                "titre": titre,
                "texte": texte,
            }
            for idx, (_, titre, texte) in enumerate(textes)
        ],
        languages="fr",
        builder=builder,
    )
    with open(outdir / "index.json", "wt", encoding="utf-8") as outfh:
        json.dump(index.serialize(), outfh, indent=2, ensure_ascii=False)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-o",
        "--outdir",
        help="Repertoire destination pour l'index",
        type=Path,
        default="export/_idx",
    )
    parser.add_argument(
        "indirs",
        help="Repertoires avec les fichiers extraits (un par ville)",
        type=Path,
        nargs="+",
    )
    return parser


def main(args: argparse.Namespace):
    """Construire un index sur des fichiers JSON"""
    index(args.indirs, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    # Done by top-level alexi if not running this as script
    parser.add_argument(
        "-v", "--verbose", help="Notification plus verbose", action="store_true"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    main(args)
