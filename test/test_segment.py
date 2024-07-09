import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

from alexi.segment import Segmenteur, detokenize, retokenize

TESTDATADIR = Path(__file__).parent / "data"
DATADIR = Path(__file__).parent.parent / "data"


def test_segment():
    with open(TESTDATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        with tempfile.TemporaryFile("w+t") as testfh:
            writer = csv.DictWriter(testfh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for word in reader:
                del word["segment"]
                writer.writerow(word)
            testfh.seek(0, 0)
            seg = Segmenteur(TESTDATADIR / "model.gz")
            reader = csv.DictReader(testfh)
            words = list(seg(reader))
    assert len(words) > 0
    with open(TESTDATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)
        for ref, hyp in zip(ref_words, words):
            ref_tag = ref["segment"].partition("-")[0]
            hyp_tag = hyp["segment"].partition("-")[0]
            print(ref["segment"], hyp["segment"], ref["text"])
            assert ref_tag == hyp_tag


@dataclass
class MockEncoding:
    """Faux sortie de faux tokenisateur"""

    tokens: list[str]
    ids: list[int]


class MockTokenizer:
    """Faux tokenisateur."""

    def encode(self, text: str, *_args, **_kwargs):
        if len(text) > 4:
            return MockEncoding(tokens=[("▁" + text[0:4]), text[4:]], ids=[333, 999])
        else:
            return MockEncoding(tokens=["▁" + text], ids=[668])


def test_retokenize():
    with open(DATADIR / "zonage_titre.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        iobs = list(reader)
        tokenizer = MockTokenizer()
        retokenized = retokenize(iobs, tokenizer)
        assert iobs != retokenized
        detokenized = list(detokenize(retokenized, tokenizer))
        assert iobs == detokenized
