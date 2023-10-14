from dataclasses import dataclass
from pdfplumber.utils.geometry import T_bbox, merge_bboxes
from typing import Any, Optional

T_obj = dict[str, Any]


@dataclass
class Bloc:
    """Élément de présentation (bloc de texte ou image)"""

    type: str
    contenu: list[T_obj]
    _bbox: Optional[T_bbox] = None
    _page_number: Optional[int] = None

    def __hash__(self):
        if self._bbox:
            return hash((self.type, self._bbox, self._page_number))
        else:
            return hash((self.type, self.contenu))

    @property
    def texte(self) -> str:
        """Représentation textuel du bloc."""
        return " ".join(x["text"] for x in self.contenu)

    @property
    def page_number(self) -> int:
        """Numéro de page de ce bloc."""
        if self._page_number is not None:
            return self._page_number
        return self.contenu[0]["page"]

    @property
    def bbox(self) -> T_bbox:
        if self._bbox is not None:
            return self._bbox
        return merge_bboxes(
            (int(word["x0"]), int(word["top"]), int(word["x1"]), int(word["bottom"]))
            for word in self.contenu
        )

    @property
    def img(self) -> str:
        bbox = ",".join(str(round(x)) for x in self.bbox)
        return f"page{self.page_number}-{bbox}.png"
