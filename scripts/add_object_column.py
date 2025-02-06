"""Ajouter une colonne avec la sortie du modele de reconnaissance
d'objets dans les CSV d'entrainement"""

import argparse
import csv
import itertools
from operator import attrgetter
from pathlib import Path

import numpy as np
from pdfplumber.utils.geometry import obj_to_bbox

from alexi import segment
from alexi.convert import FIELDNAMES
from alexi.recognize import Objets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-O",
        "--object-model",
        choices=["docling", "yolo"],
        default="docling",
        help="Modele pour detection d'objects",
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("csv", type=argparse.FileType("rt"))
    parser.add_argument("out", type=argparse.FileType("wt"))
    args = parser.parse_args()

    obj = Objets.byname(args.object_model)()

    reader = csv.DictReader(args.csv)
    fieldnames = FIELDNAMES[:]
    fieldnames.insert(0, "doclaynet")
    writer = csv.DictWriter(args.out, fieldnames, extrasaction="ignore")
    writer.writeheader()
    for (page_number, blocs), words in zip(
        itertools.groupby(obj(args.pdf), attrgetter("page_number")),
        segment.split_pages(reader),
    ):
        blocs = list(blocs)
        labels = [b.type for b in blocs]
        boxes = np.array([b.bbox for b in blocs])

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
            assert w["page"] == str(page_number)
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
            w["doclaynet"] = f"{iob}-{labels[label]}" if label is not None else "O"
            writer.writerow(w)


if __name__ == "__main__":
    main()
