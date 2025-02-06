import argparse
import csv
import logging
import re
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Union

import numpy as np
from huggingface_hub import hf_hub_download  # type: ignore
from pdfplumber.utils.geometry import obj_to_bbox
from pypdfium2 import PdfDocument, PdfPage  # type: ignore
from ultralytics import YOLO  # type: ignore

from alexi import segment
from alexi.analyse import Bloc
from alexi.convert import FIELDNAMES
from alexi.recognize import Objets

LOGGER = logging.getLogger(Path(__file__).stem)


def scale_to_model(page: PdfPage, modeldim: float):
    """Find scaling factor for model dimension."""
    maxdim = max(page.get_width(), page.get_height())
    return modeldim / maxdim


LABELMAP = {
    "Table": "Tableau",
    "Picture": "Figure",
}


class ObjetsYOLO(Objets):
    """Détecteur d'objects textuels utilisant YOLOv8 (pré-entraîné sur
    DocLayNet mais d'autres seront possibles).
    """

    def __init__(self, yolo_weights: Union[str, PathLike, None] = None):
        if yolo_weights is None:
            yolo_weights = hf_hub_download(
                repo_id="DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
                filename="yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
            )
        self.model = YOLO(str(yolo_weights))

    def __call__(
        self, pdf_path: Union[str, PathLike], pages: Union[None, Iterable[int]] = None
    ) -> Iterator[Bloc]:
        """Extraire les rectangles correspondant aux objets"""
        pdf_path = Path(pdf_path)
        pdf = PdfDocument(pdf_path)
        if pages is None:
            pages = range(1, len(pdf) + 1)
        for page_number in pages:
            page = pdf[page_number - 1]
            # FIXME: get the model input size from the model
            image = page.render(scale=scale_to_model(page, 640)).to_pil()
            results = self.model(
                source=image,
                # show_labels=True,
                # show_boxes=True,
                # show_conf=True,
            )
            assert len(results) == 1
            entry = results[0]

            # Probably should do some kind of spatial indexing
            def boxsort(e):
                """Sort by topmost-leftmost-tallest-widest."""
                _, b = e
                return (b[1], b[0], -(b[3] - b[1]), -(b[2] - b[0]))

            if len(entry.boxes.xyxy) == 0:
                continue
            ordering, box_list = zip(
                *sorted(
                    enumerate(bbox.cpu().numpy() for bbox in entry.boxes.xyxy),
                    key=boxsort,
                )
            )
            labels = [entry.names[entry.boxes.cls[idx].item()] for idx in ordering]
            img_height, img_width = entry.orig_shape
            page_width = page.get_width()
            page_height = page.get_height()
            LOGGER.info("scale x %f", page_width / img_width)
            LOGGER.info("scale y %f", page_height / img_height)
            boxes = np.array(box_list)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * page_width / img_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * page_height / img_height
            for label, box in zip(labels, boxes):
                if label in LABELMAP:
                    yield Bloc(
                        type=LABELMAP[label],
                        contenu=[],
                        _page_number=page_number,
                        _bbox=tuple(box.round()),
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_or_png", type=Path)
    parser.add_argument("csv", type=argparse.FileType("rt"))
    parser.add_argument("out", type=argparse.FileType("wt"))
    args = parser.parse_args()

    yolo_model = hf_hub_download(
        repo_id="DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
        filename="yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
    )
    docseg_model = YOLO(yolo_model)

    if args.pdf_or_png.exists():
        pdf = PdfDocument(args.pdf_or_png)
        images = (page.render(scale=scale_to_model(page, 640)).to_pil() for page in pdf)
    else:
        pngdir = args.pdf_or_png.parent
        pngre = re.compile(re.escape(args.pdf_or_png.name) + r"-(\d+)\.png")
        pngs = []
        for path in pngdir.iterdir():
            m = pngre.match(path.name)
            if m is None:
                continue
            pngs.append((int(m.group(1)), path))
        images = (path for _idx, path in sorted(pngs))

    reader = csv.DictReader(args.csv)
    fieldnames = FIELDNAMES[:]
    fieldnames.insert(0, "yolo")
    writer = csv.DictWriter(args.out, fieldnames, extrasaction="ignore")
    writer.writeheader()
    for image, words in zip(images, segment.split_pages(reader)):
        results = docseg_model(
            source=image,
            show_labels=True,
            show_boxes=True,
            show_conf=True,
        )
        assert len(results) == 1
        entry = results[0]

        # Probably should do some kind of spatial indexing
        def boxsort(e):
            """Sort by topmost-leftmost-tallest-widest."""
            _, b = e
            return (b[1], b[0], -(b[3] - b[1]), -(b[2] - b[0]))

        if len(entry.boxes.xyxy) == 0:
            continue
        ordering, boxes = zip(
            *sorted(
                enumerate(bbox.cpu().numpy() for bbox in entry.boxes.xyxy),
                key=boxsort,
            )
        )
        labels = [entry.names[entry.boxes.cls[idx].item()] for idx in ordering]
        page_width, page_height = float(words[0]["page_width"]), float(
            words[0]["page_height"]
        )
        img_height, img_width = entry.orig_shape
        print("scale x", page_width / img_width)
        print("scale y", page_height / img_height)
        boxes = np.array(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * page_width / img_width
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * page_height / img_height
        for label, box in zip(labels, boxes):
            print(label, box)

        # Boxes are (luckily) in the same coordinate system as
        # pdfplumber. But... YOLO leaves off little bits of text
        # particularly at the right edge, so we can't rely on
        # containment to match them to words
        def totally_contained(boxes, bbox):
            return (
                (bbox[[0, 1]] >= boxes[:, [0, 1]]) & (bbox[[2, 3]] <= boxes[:, [2, 3]])
            ).all(1)

        def mostly_contained(boxes, bbox):
            """Calculate inverse ratio of bbox to its intersection with each box."""
            # FIXME: This assumes boxes are normalized...
            # print("box", bbox)
            # print("boxes", boxes)
            top_left = np.maximum(bbox[[0, 1]], boxes[:, [0, 1]])
            bottom_right = np.minimum(bbox[[2, 3]], boxes[:, [2, 3]])
            intersection = np.hstack((top_left, bottom_right))
            # print("intersections", intersection)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            assert area >= 0
            width = np.maximum((intersection[:, 2] - intersection[:, 0]), 0)
            height = np.maximum((intersection[:, 3] - intersection[:, 1]), 0)
            # print("width", width)
            # print("height", height)
            iarea = width * height
            # print("area", area)
            # print("iarea", iarea)
            return iarea / area

        prev_label = None
        for w in words:
            bbox = np.array([float(w) for w in obj_to_bbox(w)])
            ratio = mostly_contained(boxes, bbox)
            # print("ratio", ratio)
            in_box = ratio.argmax()
            in_ratio = ratio.max()
            # print("in_box", in_box, "in_ratio", in_ratio)
            # in_labels = [labels[idx] for idx, val in enumerate(ratio) if val > 0.5]
            label = in_box if in_ratio > 0.5 else None
            iob = "B" if label != prev_label else "I"
            prev_label = label
            w["yolo"] = f"{iob}-{labels[label]}" if label is not None else "O"
            writer.writerow(w)


if __name__ == "__main__":
    main()
