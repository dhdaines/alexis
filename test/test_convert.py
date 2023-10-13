import csv
from pathlib import Path

from alexi.convert import Converteur, bbox_contains
from pdfplumber.utils.geometry import obj_to_bbox

DATADIR = Path(__file__).parent / "data"


def test_convert():
    with open(DATADIR / "pdf_structure.pdf", "rb") as infh:
        conv = Converteur(infh)
        words = list(conv.extract_words())
    assert len(words) > 0
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)


def test_extract_tables_and_figures():
    with open(DATADIR / "pdf_figures.pdf", "rb") as infh:
        conv = Converteur(infh)
        words = list(conv.extract_words())
        tables = list(conv.extract_tables())
        figures = list(conv.extract_figures())
        assert len(tables) == 1
        assert len(figures) == 1
        for w in words:
            if bbox_contains(tables[0].bbox, obj_to_bbox(w)):
                assert "Table" in w["tagstack"]
            if bbox_contains(figures[0].bbox, obj_to_bbox(w)):
                assert "Figure" in w["tagstack"]


if __name__ == "__main__":
    test_extract_tables_and_figures()
