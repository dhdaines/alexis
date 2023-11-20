#!/usr/bin/env python3

"""
Télécharger juste les documents dont on a besoin.
"""

import argparse
import json
import logging
import re
import subprocess
import urllib
from pathlib import Path

from bs4 import BeautifulSoup

LOGGER = logging.getLogger("download")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments to the argparse"""
    parser.add_argument(
        "-u",
        "--url",
        help="URL pour chercher les documents",
        default="https://ville.sainte-adele.qc.ca/publications.php",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Repertoire pour téléchargements",
        default="download",
        type=Path,
    )
    parser.add_argument(
        "-x",
        "--exclude",
        help="Expressions régulières pour exclure des documents",
        action="append",
        default=[],
    )
    parser.add_argument(
        "section",
        help="Expression régulière pour sélectionner la section des documents",
        default=r"urbanisme",
        nargs="?",
    )
    return parser


def main(args):
    u = urllib.parse.urlparse(args.url)
    LOGGER.info("Downloading %s", args.url)
    try:
        subprocess.run(
            [
                "wget",
                "--quiet",
                "--no-check-certificate",
                "--timestamping",
                "-P",
                str(args.outdir),
                args.url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as err:
        if err.returncode != 8:
            raise
    excludes = [re.compile(r) for r in args.exclude]
    paths = []
    with open(args.outdir / Path(u.path).name) as infh:
        soup = BeautifulSoup(infh, "lxml")
        for h2 in soup.find_all("h2", string=re.compile(args.section, re.I)):
            ul = h2.find_next("ul")
            for li in ul.find_all("li"):
                path = li.a["href"]
                excluded = False
                for r in excludes:
                    if r.search(path):
                        excluded = True
                        break
                if not excluded:
                    paths.append(path)
    urls = {}
    for p in paths:
        up = urllib.parse.urlparse(p)
        if up.netloc:
            url = p
        else:
            url = f"{u.scheme}://{u.netloc}{up.path}"
        urls[Path(up.path).name] = url
    if not urls:
        LOGGER.error("Could not find any documents to download!")
        return
    for u in urls.values():
        LOGGER.info("Downloading %s", u)
    try:
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "--timestamping",
                "--quiet",
                "-P",
                str(args.outdir),
                *urls.values(),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as err:
        if err.returncode != 8:
            raise
    with open(args.outdir / "index.json", "wt") as outfh:
        json.dump(urls, outfh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    main(args)
