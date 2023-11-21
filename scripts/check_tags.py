"""Vérifier les annotations."""

import argparse
import csv
from pathlib import Path
from alexi.segment import Bullet
from alexi.analyse import group_iob, Bloc


def make_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV", type=Path)
    return parser


def check_bloc(bloc: Bloc, lineno: int) -> None:
    prev_bio = "O"
    prev_segment = ""
    errors = []
    for idx, word in enumerate(bloc.contenu):
        bio, _, segment = word.get("segment", "O").partition("-")
        if bio == "I" and prev_bio == "I" and segment != prev_segment:
            errors.append("%d: I-%s => I-%s" % (lineno + idx, prev_segment, segment))
        if bio == "I" and segment == "Liste":
            for pattern in Bullet:
                if pattern.value.match(word["text"]):
                    errors.append(
                        "%d: I-Liste Bullet: %s"
                        % (
                            lineno + idx,
                            word["text"],
                        )
                    )
                    break
        prev_bio = bio
        prev_segment = segment
    return errors


def main(args: argparse.Namespace) -> None:
    for path in args.csvs:
        lineno = 2
        with open(path, "rt") as infh:
            reader = csv.DictReader(infh)
            for bloc in group_iob(reader):
                errors = check_bloc(bloc, lineno)
                if errors:
                    print(
                        "%s page %d lines %d-%d:\n%s: %s"
                        % (
                            path.name,
                            bloc.page_number,
                            lineno,
                            lineno + len(bloc.contenu),
                            bloc.type,
                            bloc.texte,
                        )
                    )
                    for e in errors:
                        print("\t%s" % e)
                lineno += len(bloc.contenu)


if __name__ == "__main__":
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
