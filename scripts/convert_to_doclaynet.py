"""
Convertir les annotations ALEXI en annotations DocLayNet pour évaluation.
"""

import argparse
import csv

EQUIVS = {
    "Pied": "Page-footer",
    "Tete": "Page-header",
    "Liste": "List-item",
    "Chapitre": "Section-header",
    "Section": "Section-header",
    "SousSection": "Section-header",
    "Chapitre": "Section-header",
    "Annexe": "Section-header",
    "Article": "Section-header",
    "Alinea": "Text",
}
IGNORES = {
    "Footnote",
    "Formula",
    "Title",
    "Caption",
    "Table",
    "Picture",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "infile", type=argparse.FileType("rt"), help="Fichier CSV input"
    )
    parser.add_argument(
        "outfile", type=argparse.FileType("wt"), help="Fichier CSV output"
    )
    args = parser.parse_args()
    reader = csv.DictReader(args.infile)
    writer = csv.DictWriter(args.outfile, fieldnames=["yolo", *reader.fieldnames])
    writer.writeheader()
    for word in reader:
        iob, sep, tag = word["segment"].partition("-")
        if sep and tag in EQUIVS:
            word["yolo"] = f"{iob}-{EQUIVS[tag]}"
        else:  # nothing else is YOLO
            word["yolo"] = "O"
        writer.writerow(word)


if __name__ == "__main__":
    main()
