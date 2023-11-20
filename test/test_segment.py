import csv
import tempfile
from pathlib import Path

from alexi.segment import Segmenteur

DATADIR = Path(__file__).parent / "data"


def test_segment():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        with tempfile.TemporaryFile("w+t") as testfh:
            writer = csv.DictWriter(testfh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for word in reader:
                del word["segment"]
                writer.writerow(word)
            testfh.seek(0, 0)
            seg = Segmenteur()
            reader = csv.DictReader(testfh)
            words = list(seg(reader))
    assert len(words) > 0
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)
        for ref, hyp in zip(ref_words, words):
            ref_tag = ref["segment"].partition("-")[0]
            hyp_tag = hyp["segment"].partition("-")[0]
            print(ref["segment"], hyp["segment"], ref["text"])
            assert ref_tag == hyp_tag
