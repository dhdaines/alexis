from pathlib import Path

from alexi.extract import Extracteur

DATADIR = Path(__file__).parent / "data"
TRAINDIR = Path(__file__).parent.parent / "data"


def test_extracteur(tmp_path: Path):
    extracteur = Extracteur(tmp_path)
    doc = extracteur(DATADIR / "zonage_zones.pdf")
    docdir = tmp_path / "zonage_zones"
    assert (docdir / "img").is_dir()
    extracteur.output_doctree([doc])
    assert (tmp_path / "index.html").exists()
    extracteur.output_html(doc)
    assert (docdir / "index.html").exists()
    assert (docdir / "Article" / "index.html").exists()
    for article in (431, 432, 444, 454, 464, 465, 474):
        assert (docdir / "Article" / str(article) / "index.html").exists()
    assert (docdir / "Chapitre" / "index.html").exists()
    assert (docdir / "Chapitre" / "7" / "index.html").exists()
    for sec in range(1, 4):
        assert (
            docdir / "Chapitre" / "7" / "Section" / str(sec) / "index.html"
        ).exists()
        subsec = list(
            (docdir / "Chapitre" / "7" / "Section" / str(sec) / "SousSection").iterdir()
        )
        assert len(subsec) == 2
