import logging
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Union

import numpy as np
from huggingface_hub import hf_hub_download  # type: ignore
from pypdfium2 import PdfDocument, PdfPage  # type: ignore
from ultralytics import YOLO  # type: ignore

from alexi.analyse import Bloc
from alexi.recognize import Objets

LOGGER = logging.getLogger(Path(__file__).stem)


def scale_to_model(page: PdfPage, modeldim: float):
    """Find scaling factor for model dimension."""
    maxdim = max(page.get_width(), page.get_height())
    return modeldim / maxdim


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
        self,
        pdf_path: Union[str, PathLike],
        pages: Union[None, Iterable[int]] = None,
        labelmap: Union[dict, None] = None,
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
                if labelmap is None or label in labelmap:
                    yield Bloc(
                        type=(label if labelmap is None else labelmap[label]),
                        contenu=[],
                        _page_number=page_number,
                        _bbox=tuple(box.round()),
                    )
            page.close()
        pdf.close()
