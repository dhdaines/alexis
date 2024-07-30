"""Télécharger les lois et règlements référenciés de LegisQuébec.

Notez que, en théorie, la redistribution et même le "téléchargement"
de ces fichiers est interdit, ce qui a très peu de sens étant donné
qu'il faut les télécharger pour les consulter.
"""

import argparse
import httpx
import json
import logging
import urllib
import time

from pathlib import Path

LOGGER = logging.getLogger(Path(__file__).stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--outdir", help="Repertoire sortie", default="legisquebec", type=Path
    )
    parser.add_argument("metadata", help="Fichier JSON avec metadonnées", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.metadata, "rt") as infh:
        metadata = json.load(infh)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for url in metadata["urls"]:
        parts = urllib.parse.urlparse(url)
        path = Path(urllib.parse.unquote(parts.path))
        # Do not use with_suffix because it sucks
        outfile = args.outdir / (str(path.name).strip() + ".html")
        if outfile.exists():
            continue
        LOGGER.info("Downloading %s to %s", url, outfile)
        try:
            r = httpx.get(url)
        except httpx.ReadTimeout:
            # HACK! (possibly httpx can do this?)
            time.sleep(5)
            r = httpx.get(url)
        if r.status_code == 200:
            with open(outfile, "wt") as outfh:
                outfh.write(r.text)
        else:
            LOGGER.error("Error: %s", r)

        url = url.replace("/document/", "/pdf/") + ".pdf"
        outfile = args.outdir / (str(path.name).strip() + ".pdf")
        LOGGER.info("Downloading %s to %s", url, outfile)
        try:
            r = httpx.get(url)
        except httpx.ReadTimeout:
            time.sleep(5)
            r = httpx.get(url)
        if r.status_code == 200:
            with open(outfile, "wb") as outfh:
                outfh.write(r.content)
        else:
            LOGGER.error("Error: %s", r)


if __name__ == "__main__":
    main()
