import itertools
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from operator import itemgetter
from pathlib import Path
from typing import Optional

import pytest
from alexi import extract_main
from alexi.convert import Converteur
from alexi.label import Classificateur
from alexi.segment import Segmenteur
from alexi.types import Reglement

TOPDIR = Path(__file__).parent.parent


@dataclass
class ExtractArgs:
    pdf: Optional[Path] = None
    images: Optional[Path] = None
    pages: Optional[str] = None


EXPECT_TITLES = [
    ("data/train/00-Reglement-1000-2008-PPC.pdf", "REGLEMENT NO. 1000-2008-PPC"),
    (
        "data/train/2022-04-19-Rgl-1324-redevances-adopte.pdf",
        "RÈGLEMENT 1324\nrelatif au paiement d’une contribution pour financer en\ntout ou en partie une dépense liée à l’ajout,\nl’agrandissement ou la modification d’infrastructures ou\nd’équipements municipaux",
    ),
    (
        "data/train/Rgl-1314-2021-L-Lotissement.pdf",
        "Règlement de lotissement Numéro 1314-2021-L",
    ),
    (
        "data/train/Rgl-1314-2021-PC-version-en-vigueur-20230509.pdf",
        "Règlement sur les permis et certificats Numéro 1314-2021-PC",
    ),
    (
        "data/train/Rgl-1314-2021-TM-Travaux-municipaux.pdf",
        "Règlement sur les ententes relatives à des travaux municipaux Numéro 1314-2021-TM",
    ),
    (
        "data/train/xx-2020-04-20-RGL-1289-Formation-CCE-adopte_1.pdf",
        "RÈGLEMENT 1289\nconcernant formation d’un Comité consultatif en\nenvironnement",
    ),
]


@pytest.mark.parametrize("pdf,title", EXPECT_TITLES)
def test_extract_titles(pdf, title):
    with redirect_stdout(StringIO()) as out:
        extract_main(ExtractArgs(pdf=TOPDIR / pdf, pages="1,2"))
    reg = Reglement.model_validate_json(out.getvalue())
    assert reg.titre == title


def test_extract_toc():
    converteur = Converteur()
    segmenteur = Segmenteur()
    classificateur = Classificateur()
    doc = converteur("data/train/Rgl-1314-2021-L-Lotissement.pdf", range(5))
    doc = segmenteur(doc)
    doc = classificateur(doc)
    for page, words in itertools.groupby(doc, itemgetter("page")):
        found_toc = any(w["tag"] in ("B-TOC", "I-TOC") for w in words)
        if page == 1 or page == 5:
            assert not found_toc
        else:
            assert found_toc


EXPECT_CHAPTERS = [
    ("data/train/00-Reglement-1000-2008-PPC.pdf", None, 3),
    ("data/train/2022-04-19-Rgl-1324-redevances-adopte.pdf", None, 0),
    ("data/train/Rgl-1314-2021-L-Lotissement.pdf", "1,5,6,14,15,18,19,23,24,26", 4),
    (
        "data/train/Rgl-1314-2021-PC-version-en-vigueur-20230509.pdf",
        "1,8,16,20,27,43,56,60",
        7,
    ),
    ("data/train/Rgl-1314-2021-TM-Travaux-municipaux.pdf", "1,4,9", 2),
    ("data/train/xx-2020-04-20-RGL-1289-Formation-CCE-adopte_1.pdf", None, 0),
]


@pytest.mark.parametrize("pdf,pages,nchap", EXPECT_CHAPTERS)
def test_extract_chapitres(pdf, pages, nchap):
    with redirect_stdout(StringIO()) as out:
        extract_main(ExtractArgs(pdf=TOPDIR / pdf, pages=pages))
    reg = Reglement.model_validate_json(out.getvalue())
    assert len(reg.chapitres) == nchap
