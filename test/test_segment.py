import csv
from pathlib import Path

from alexi.segment import Segmenteur

DATADIR = Path(__file__).parent / "data"


def test_convert():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        conv = Segmenteur()
        words = list(conv(infh))
    assert len(words) > 0
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)
        for ref, hyp in zip(ref_words, words):
            assert ref["tag"] == hyp["tag"]
