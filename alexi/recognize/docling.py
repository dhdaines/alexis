"""Analyse de mise en page avec modèle RT-DETR de IBM (Docling)"""

import logging
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Union

from docling_ibm_models.layoutmodel.layout_predictor import (  # type: ignore
    LayoutPredictor,
)
from pypdfium2 import PdfDocument, PdfPage  # type: ignore

from alexi.analyse import Bloc
from alexi.recognize import Objets

LOGGER = logging.getLogger(Path(__file__).stem)
LABELMAP = {
    "Table": "Tableau",
    "Picture": "Figure",
}


def scale_to_model(page: PdfPage, modeldim: float):
    """Find scaling factor for model dimension."""
    mindim = min(page.get_width(), page.get_height())
    return modeldim / mindim


def load_model_from_hub() -> Path:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        "ds4sd/docling-models", "model_artifacts/layout/preprocessor_config.json"
    )
    hf_hub_download("ds4sd/docling-models", "model_artifacts/layout/config.json")
    weights_path = hf_hub_download(
        "ds4sd/docling-models", "model_artifacts/layout/model.safetensors"
    )
    return Path(weights_path).parent


class ObjetsDocling(Objets):
    """Détecteur d'objects textuels utilisant RT-DETR."""

    def __init__(
        self,
        model_path: Union[str, PathLike, None] = None,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
    ) -> None:
        if model_path is None:
            model_path = load_model_from_hub()
        else:
            model_path = Path(model_path)
        self.model = LayoutPredictor(
            str(model_path),
            device=device,
            num_threads=num_threads,
            base_threshold=base_threshold,
        )
        self.model_info = self.model.info()

    def __call__(
        self, pdf_path: Union[str, PathLike], pages: Union[None, Iterable[int]] = None
    ) -> Iterator[Bloc]:
        pdf_path = Path(pdf_path)
        pdf = PdfDocument(pdf_path)
        if pages is None:
            pages = range(1, len(pdf) + 1)
        for page_number in pages:
            page = pdf[page_number - 1]
            scale = scale_to_model(page, self.model_info["image_size"])
            image = page.render(scale=scale).to_pil()

            def boxsort(box):
                """Sort by topmost-leftmost-tallest-widest."""
                return (
                    box["t"],
                    box["l"],
                    -(box["b"] - box["t"]),
                    -(box["r"] - box["l"]),
                )

            boxes = sorted(self.model.predict(image), key=boxsort)
            for box in boxes:
                if box["label"] in LABELMAP:
                    bbox = tuple(
                        round(x / scale)
                        for x in (box["l"], box["t"], box["r"], box["b"])
                    )
                    yield Bloc(
                        type=LABELMAP[box["label"]],
                        contenu=[],
                        _page_number=page_number,
                        _bbox=bbox,
                    )
        pdf.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    recx = ObjetsDocling()
    for bloc in recx(args.pdf):
        print(bloc)


if __name__ == "__main__":
    main()
