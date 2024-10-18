#!/usr/bin/env python3

"""
Télécharger juste les documents dont on a besoin.
"""

import argparse
import asyncio
import datetime
import email.utils
import httpx
import json
import logging
import os
import re
import ssl
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup

CONTEXT = ssl.create_default_context()
# Work around misconfigured Sainte-Adèle (and maybe others eventually)
# website by adding some certificates to the default ones
CONTEXT.load_verify_locations(Path(__file__).parent / "extracerts.pem")
CLIENT = httpx.AsyncClient(verify=CONTEXT)
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
        "--all-pdf-links",
        action="store_true",
        help="Télécharger les liens vers des PDF dans le document "
        "sans égard à sa structure",
    )
    parser.add_argument(
        "section",
        help="Expression régulière pour sélectionner la section des documents",
        default=r"urbanisme",
        nargs="?",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    u = urllib.parse.urlparse(args.url)
    index = args.outdir / "index.html"
    LOGGER.info("Downloading %s to %s", args.url, index)
    args.outdir.mkdir(parents=True, exist_ok=True)
    r = await CLIENT.get(args.url, follow_redirects=True)
    if r.status_code != 200:
        LOGGER.error("Download failed: %s", r)
        return
    with open(index, "w") as outfh:
        outfh.write(r.text)
    excludes = [re.compile(r) for r in args.exclude]
    paths = []
    with open(index) as infh:
        soup = BeautifulSoup(infh, "lxml")
        if args.all_pdf_links:
            for a in soup.find_all("a"):
                if "href" not in a.attrs:
                    continue
                path = a["href"]
                if path.lower().endswith(".pdf"):
                    paths.append(path)
        else:
            for h2 in soup.find_all("h2", string=re.compile(args.section, re.I)):
                ul = h2.find_next("ul")
                for li in ul.find_all("li"):
                    paths.append(li.a["href"])
    urls = {}
    for p in paths:
        excluded = False
        for r in excludes:
            if r.search(p):
                excluded = True
                break
        if excluded:
            continue
        up = urllib.parse.urlparse(p)
        if up.netloc:
            url = p
        else:
            url = f"{u.scheme}://{u.netloc}{up.path}"
        outname = Path(up.path).name
        outpath = args.outdir / outname
        try:
            mtime = datetime.datetime.fromtimestamp(
                outpath.stat().st_mtime, tz=datetime.timezone.utc
            )
            urls[outname] = {
                "url": url,
                "modified": mtime.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            }
        except FileNotFoundError:
            urls[outname] = {"url": url}
    if not urls:
        LOGGER.error("Could not find any documents to download!")
        return
    for outname, info in urls.items():
        if "modified" in info:
            r = await CLIENT.get(
                info["url"], headers={"if-modified-since": info["modified"]}
            )
            if r.status_code == 304:
                continue
        else:
            r = await CLIENT.get(info["url"])
        with open(args.outdir / outname, "wb") as outfh:
            outfh.write(r.content)
        if "last-modified" in r.headers:
            info["modified"] = r.headers["last-modified"]
            # DO NOT USE STRPTIME OMG WTF
            mtime = email.utils.parsedate_to_datetime(
                r.headers["last-modified"]
            ).timestamp()
            os.utime(args.outdir / outname, (mtime, mtime))

    with open(args.outdir / "index.json", "wt") as outfh:
        json.dump(urls, outfh, indent=2)


def main(args: argparse.Namespace) -> None:
    asyncio.run(async_main(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    asyncio.run(async_main(args))
