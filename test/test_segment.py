import csv
import tempfile
from pathlib import Path

from alexi.segment import Segmenteur, group_paragraphs

DATADIR = Path(__file__).parent / "data"


def test_segment():
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        with tempfile.TemporaryFile("w+t") as testfh:
            writer = csv.DictWriter(testfh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for word in reader:
                del word["tag"]
                writer.writerow(word)
            testfh.seek(0, 0)
            seg = Segmenteur()
            words = list(seg(testfh))
    assert len(words) > 0
    with open(DATADIR / "pdf_structure.csv", "rt") as infh:
        reader = csv.DictReader(infh)
        ref_words = list(reader)
        assert len(words) == len(ref_words)
        for ref, hyp in zip(ref_words, words):
            ref_tag = ref["tag"].partition("-")[0]
            hyp_tag = hyp["tag"].partition("-")[0]
            print(ref["tag"], hyp["tag"], ref["text"])
            assert ref_tag == hyp_tag


def test_group_paragaraphs():
    TESTWORDS = [
        {"tag": "O"},
        {"tag": "B-Foo"},
        {"tag": "I-Foo"},
        {"tag": "O"},
        {"tag": "O"},
        {"tag": "O"},
        {"tag": "B-Foo"},
        {"tag": "I-Foo"},
        {"tag": "I-Foo"},
        {"tag": "O"},
        {"tag": "B-Bar"},
        {"tag": "B-Bar"},
        {"tag": "O"},
    ]
    groups = list(group_paragraphs(TESTWORDS))
    assert groups == [
        (
            "O",
            [
                {"tag": "O"},
            ],
        ),
        (
            "Foo",
            [
                {"tag": "B-Foo"},
                {"tag": "I-Foo"},
            ],
        ),
        (
            "O",
            [
                {"tag": "O"},
                {"tag": "O"},
                {"tag": "O"},
            ],
        ),
        (
            "Foo",
            [
                {"tag": "B-Foo"},
                {"tag": "I-Foo"},
                {"tag": "I-Foo"},
            ],
        ),
        (
            "O",
            [
                {"tag": "O"},
            ],
        ),
        (
            "Bar",
            [
                {"tag": "B-Bar"},
            ],
        ),
        (
            "Bar",
            [
                {"tag": "B-Bar"},
            ],
        ),
        (
            "O",
            [
                {"tag": "O"},
            ],
        ),
    ]
