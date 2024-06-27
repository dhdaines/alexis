"""Refaire la tokenisation en alignant les etiquettes et traits sur
les nouveaux segments."""

# To implement:
#
# - retokenize: CSV/IOB -> CSV/IOB
# - detokenize: CSV/IOB -> CSV/IOB
#
# This should improve existing CRF as well!


import argparse
import csv
import sys
import tokenizers

from alexi.convert import write_csv
from alexi.segment import retokenize, detokenize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-t", "--tokenizer", help="Nom du tokenisateur", default="camembert-base"
    )
    parser.add_argument(
        "-d", "--detokenize", help="Inverser la retokenisation", action="store_true"
    )
    parser.add_argument(
        "csv",
        help="Fichier CSV à traiter",
        type=argparse.FileType("rt"),
    )
    args = parser.parse_args()
    tokenizer = tokenizers.Tokenizer.from_pretrained(args.tokenizer)
    reader = csv.DictReader(args.csv)
    if args.detokenize:
        write_csv(detokenize(reader, tokenizer), sys.stdout)
    else:
        write_csv(retokenize(reader, tokenizer), sys.stdout)


if __name__ == "__main__":
    main()
