"""
Construire un index pour faire des recherches dans les données extraites.
"""

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass

from bs4 import BeautifulSoup
from lunr import lunr, get_default_builder, trimmer
from lunr.pipeline import Pipeline
from unidecode import unidecode

LOGGER = logging.getLogger("index")


@dataclass
class Document:
    url: str
    titre: str
    texte: str


def body_text(soup: BeautifulSoup):
    body = soup.div(id="body")[0]
    for header in body(class_="header"):
        header.extract()
    for img in body("img"):
        alt = soup.new_tag("p")
        alt.string = img["alt"]
        img.replace_with(alt)
    return re.sub("\n\n+", "\n\n", soup.text.strip())


def unifold(token, _idx=None, _tokens=None):
    def wrap_unidecode(text, _metadata):
        return unidecode(text)

    return token.update(wrap_unidecode)


Pipeline.register_function(unifold, "unifold")


def index(indir: Path, outdir: Path) -> None:
    """
    Generer l'index a partir des fichiers HTML.
    """
    # Metadata (use to index specific zones, etc)
    # with open(indir / "index.json", "rt") as infh:
    #     metadata = json.load(infh)

    # lunr does not do storage so we store plaintext here
    textes = {}

    # Use index.html to find things (as in the js version)
    LOGGER.info("Traitement: %s", indir / "index.html")
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
        url = section.a["href"]
        # Assume it is a relative URL (we made it)
        LOGGER.info("Traitement: %s: %s", title, indir / url)
        with open(indir / url, "rt") as infh:
            subsoup = BeautifulSoup(infh, features="lxml")
            textes[url] = {"titre": title, "texte": body_text(subsoup)}
    for text in soup.select("li.leaf"):
        title = text.a.text
        url = text.a["href"]
        LOGGER.info("Traitement: %s: %s", title, indir / url)
        with open(indir / url, "rt") as infh:
            subsoup = BeautifulSoup(infh, features="lxml")
            textes[url] = {"titre": title, "texte": body_text(subsoup)}

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
    builder.metadata_whitelist.append("position")
    LOGGER.info("pipeline: %s", builder.pipeline)
    LOGGER.info("search pipeline: %s", builder.pipeline)

    index = lunr(
        ref="url",
        fields=[{"field_name": "titre", "boost": 2}, "texte"],
        documents=[
            {"url": url, "titre": doc["titre"], "texte": doc["texte"]}
            for url, doc in textes.items()
        ],
        languages="fr",
        builder=builder,
    )
    with open(outdir / "index.json", "wt", encoding="utf-8") as outfh:
        json.dump(index.serialize(), outfh, indent=2, ensure_ascii=False)
