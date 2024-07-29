import csv
from pathlib import Path

from alexi.convert import Converteur

DATADIR = Path(__file__).parent / "data"


def test_convert() -> None:
    with open(DATADIR / "pdf_structure.pdf", "rb") as infh:
        conv = Converteur(infh)
        words = list(conv.extract_words())
    assert len(words) > 0
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)


if __name__ == "__main__":
    test_convert()
