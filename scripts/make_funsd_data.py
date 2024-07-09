"""
Convert our data to FUNSD style (IOB boxes and PNG images)
"""

import argparse
import itertools
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pdfplumber
from sklearn.model_selection import KFold
from tokenizers import Tokenizer

from alexi import segment
from alexi.format import line_breaks
from alexi.types import T_obj

LOGGER = logging.getLogger(Path(__file__).stem)


def resegment(page: list[T_obj], max_seq_length: int) -> Iterator[list[T_obj]]:
    """Refaire la segmentation d'une page pour respecter les limites de tokens"""
    if len(page) < max_seq_length:
        yield page
        return
    # Find the mode of line spacing
    # Try to break where there is a larger line break
    lines = list(line_breaks(page))
    spacing = Counter(
        int(b[0]["top"]) - int(a[0]["bottom"]) for a, b in itertools.pairwise(lines)
    )
    line_gap = spacing.most_common(1)[0][0]
    LOGGER.info("line gap %d", line_gap)
    # Total count of output (for validation)
    page_length = 0
    # Length and extent of current "paragraph"
    tcount = start = cur = 0
    for idx, line in enumerate(lines):
        LOGGER.info(
            "line %d start %d cur %d tcount %d => %d",
            idx,
            start,
            cur,
            tcount,
            tcount + len(line),
        )
        tcount += len(line)
        # Try to output paragraphs, fall back to lines, then if
        # necessary fall back to sub-lines (shouldn't be necessary)
        while tcount > max_seq_length:
            if cur > start:
                seg = list(itertools.chain.from_iterable(lines[start:cur]))
                LOGGER.info("output paragraph %d:%d (%d tokens)", start, cur, len(seg))
                assert len(seg) <= max_seq_length
                page_length += len(seg)
                yield seg
                start = cur
                tcount = sum(len(x) for x in lines[start : idx + 1])
            elif idx > cur:
                seg = list(itertools.chain.from_iterable(lines[start:idx]))
                assert len(seg) <= max_seq_length
                LOGGER.info("output lines %d:%d (%d tokens)", start, idx, len(seg))
                page_length += len(seg)
                yield seg
                start = cur = idx
                tcount = len(line)
            else:
                LOGGER.warning(
                    "Very long line %d (%d tokens), must split to %d",
                    idx,
                    tcount,
                    max_seq_length,
                )
                assert idx == start
                assert idx == cur
                for start in range(0, len(line), max_seq_length):
                    seg = line[start : start + max_seq_length]
                    page_length += len(seg)
                    yield seg
                start = cur = idx + 1
                tcount = 0
        line_line_gap = int(line[0]["top"]) - int(lines[idx - 1][0]["bottom"])
        if line_line_gap > line_gap + 1:
            LOGGER.info(
                "break at line %d line gap %d",
                idx,
                line_line_gap,
            )
            cur = idx
    if start < len(lines):
        seg = list(itertools.chain.from_iterable(lines[start:]))
        assert len(seg) <= max_seq_length
        page_length += len(seg)
        yield seg
    assert page_length == len(page)


def write_fold(
    pages: Iterable[T_obj],
    tokenizer: Tokenizer,
    max_seq_length: int,
    outbase: Path,
    imgdir: Path | None = None,
) -> set[str]:
    txtpath = outbase.with_suffix(".txt")
    boxpath = txtpath.with_stem(f"{txtpath.stem}_box")
    imgpath = txtpath.with_stem(f"{txtpath.stem}_image")
    if imgdir is not None:
        imgdir.mkdir(parents=True, exist_ok=True)
    labels = set()
    path = pngpath = None
    with open(txtpath, "wt") as txtfh, open(boxpath, "wt") as boxfh, open(
        imgpath, "wt"
    ) as imgfh:
        for page in pages:
            if page[0]["path"] != path:
                path = Path(page[0]["path"])
                pageno = int(page[0]["page"])
                pdfpath = path.with_suffix(".pdf")
                if imgdir is not None:
                    pdf = pdfplumber.open(pdfpath)
                    ipage = pdf.pages[page - 1]
                    pngpath = imgdir / f"{pdfpath.stem}-{pageno}.png"
                    ipage.to_image().save(pngpath)
                else:
                    pngpath = Path(f"{pdfpath.stem}-{pageno}.png")
            page_width = int(page[0]["page_width"])
            page_height = int(page[0]["page_height"])
            maxdim = max(page_width, page_height)
            mediabox = " ".join(
                str(int(f / maxdim * 1000)) for f in (page_width, page_height)
            )
            tpage = list(segment.retokenize(page, tokenizer))
            LOGGER.info("page: %d tokens", len(tpage))
            tseg = list(resegment(tpage, max_seq_length))
            for seg in tseg:
                LOGGER.info("seg: %d tokens", len(seg))
                # Detokenize here since that's what the current training
                # code wants (even though it's going to tokenize it
                # again...)
                for word in segment.detokenize(seg, tokenizer):
                    labels.add(word["segment"])
                    bbox = [
                        int(float(word[f]) / maxdim * 1000)
                        for f in "x0 top x1 bottom".split()
                    ]
                    assert all(x <= 1000 for x in bbox)
                    box = " ".join(str(f) for f in bbox)
                    print("\t".join((word["text"], word["segment"])), file=txtfh)
                    print(
                        "\t".join((word["text"], box)),
                        file=boxfh,
                    )
                    print(
                        "\t".join((word["text"], box, mediabox, pngpath.name)),
                        file=imgfh,
                    )
                for fh in txtfh, boxfh, imgfh:
                    print(file=fh)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--outdir",
        default="data",
        help="Repertoire pour fichiers de sortie",
        type=Path,
    )
    parser.add_argument(
        "-x",
        "--cross-validation-folds",
        default=4,
        type=int,
        help="Nombre de partitions de validation croisée.",
    )
    parser.add_argument("-d", "--imgdir", help="Repertoire pour images", type=Path)
    parser.add_argument("--seed", default=1381, type=int, help="Graine aléatoire")
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Longueur maximale des sequences",
    )
    parser.add_argument("csvs", nargs="+", help="Fichiers CSV", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Need to split inside pages to respect the 512-token limit, so we
    # will tokenize to find out where we need to split
    tokenizer = Tokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    pages = list(segment.split_pages(segment.filter_tab(segment.load(args.csvs))))
    kf = KFold(n_splits=4, shuffle=True, random_state=args.seed)
    for fold, (train_idx, dev_idx) in enumerate(kf.split(pages)):
        foldir = args.outdir / f"fold{fold + 1}"
        foldir.mkdir()
        labels = write_fold(
            (pages[x] for x in train_idx),
            tokenizer,
            args.max_seq_length - 3,
            foldir / "train",
        )
        devlabels = write_fold(
            (pages[x] for x in dev_idx),
            tokenizer,
            args.max_seq_length - 3,
            foldir / "test",
        )
        labels.update(devlabels)
        with open(foldir / "labels.txt", "wt") as outfh:
            for label in labels:
                print(label, file=outfh)


if __name__ == "__main__":
    main()
