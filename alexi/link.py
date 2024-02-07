"""
Ajouter des hyperliens au HTML
"""

import argparse
import logging
import os
import re

from pathlib import Path

LOGGER = logging.getLogger("link")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments to the argparse"""
    parser.add_argument(
        "-o",
        "--outdir",
        help="Repertoire avec les documents",
        type=Path,
        default="export",
    )
    return parser


def link_file(root: Path, path: Path, paths: list[Path]) -> None:
    LOGGER.info("recherche de liens dans %s", path)
    relroot = path.relative_to(root)
    parts = relroot.parts
    temppath = path.with_suffix(".txt.tmp")
    with open(path, "rt") as infh, open(temppath, "wt") as tempfh:
        for spam in infh:
            # Do this in a very artificially-unintelligent way using
            # regular expressions for now, and not even parsing HTML
            # (FIXME: we will parse the HTML)
            if re.search(r"<a|<h|<img", spam):
                tempfh.write(spam)
                continue

            def replace_section(m):
                # Look for a corresponding section (FIXME: only works
                # for articles, chapters, and annexes for the moment)
                sectype = m.group(1).title()
                num = m.group(2).strip(".")
                artpath = root / parts[0] / sectype / num / "index.html"
                if artpath.exists():
                    relpath = os.path.relpath(artpath, path.parent)
                    LOGGER.info("%s -> %s", m.group(0), relpath)
                    return f'<a href="{relpath}">{m.group(0)}</a>'
                else:
                    LOGGER.info("%s non trouvé", m.group(0))
                    return m.group(0)

            spam = re.sub(
                r"\b(article|chapitre|section|sous-section|annexe)\s+([\d\.]+)",
                replace_section,
                spam,
                flags=re.IGNORECASE,
            )
            tempfh.write(spam)
    temppath.rename(path)


def main(args: argparse.Namespace) -> None:
    """Fonction principale."""
    # Find all the possible targets first (will determine if we can link)
    paths = []
    for root, dirs, files in os.walk(args.outdir):
        proot = Path(root)
        if proot.name in ("Article", "Section", "Chapitre", "SousSection", "Annexe"):
            LOGGER.info("non-traitement de index des %s", proot.name)
            continue
        if proot == args.outdir:
            LOGGER.info("non-traitement de repertoire principal")
            continue
        for f in files:
            paths.append(proot / f)
    for root, dirs, files in os.walk(args.outdir):
        proot = Path(root)
        if proot.name in ("Article", "Section", "Chapitre", "SousSection", "Annexe"):
            LOGGER.info("non-traitement de index des %s", proot.name)
            continue
        if proot == args.outdir:
            LOGGER.info("non-traitement de repertoire principal")
            continue
        for f in files:
            p = Path(f)
            if p.suffix != ".html":
                continue
            link_file(args.outdir, proot / f, paths)


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
