from pathlib import Path

from pdfplumber.utils.geometry import obj_to_bbox

from alexi.convert import Converteur
from alexi.recognize import Objets, bbox_contains

DATADIR = Path(__file__).parent / "data"


def test_extract_tables_and_figures() -> None:
    conv = Converteur(DATADIR / "pdf_figures.pdf")
    obj = Objets()
    words = list(conv.extract_words())
    images = list(obj(DATADIR / "pdf_figures.pdf"))
    assert len(images) == 2
    table = next(img for img in images if img.type == "Tableau")
    figure = next(img for img in images if img.type == "Figure")
    for w in words:
        if bbox_contains(table.bbox, obj_to_bbox(w)):
            assert "Table" in w["tagstack"]
        if bbox_contains(figure.bbox, obj_to_bbox(w)):
            assert "Figure" in w["tagstack"]


if __name__ == "__main__":
    test_extract_tables_and_figures()
