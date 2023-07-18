from alexi.label import group_paragraphs


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
