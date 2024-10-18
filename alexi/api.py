"""
API pour indexes ALEXI
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from lunr.index import Index  # type: ignore
from lunr.languages import get_nltk_builder  # type: ignore
from pydantic import BaseModel

from alexi.index import unifold

# This is just here to register the necessary pipeline functions
get_nltk_builder(["fr"])
LOGGER = logging.getLogger(__name__)
DOCDIR = Path(os.getenv("ALEXI_DIR", "export"))
INDEXDIR = DOCDIR / "_idx"
with open(INDEXDIR / "index.json", "rt", encoding="utf-8") as infh:
    INDEX = Index.load(json.load(infh))
INDEX.pipeline.add(unifold)
with open(INDEXDIR / "textes.json", "rt", encoding="utf-8") as infh:
    DOCS = json.load(infh)
API = FastAPI()


class SearchResult(BaseModel):
    url: str
    titre: str
    texte: str
    termes: List[str]
    score: float


@API.get("/villes")
async def villes() -> List[str]:
    return [
        path.name for path in DOCDIR.iterdir() if path.is_dir() and path.name != "_idx"
    ]


@API.get("/recherche")
async def recherche(
    q: str, v: Union[str, None] = None, limite: int = 10
) -> List[SearchResult]:
    results = []
    for r in INDEX.search(q):
        url, titre, texte = DOCS[int(r["ref"])]
        if v is None or v == "" or url.startswith(v):
            md = r["match_data"]
            results.append(
                SearchResult(
                    url=url,
                    titre=titre,
                    texte=texte,
                    termes=list(md.metadata.keys()),
                    score=r["score"],
                )
            )
            if len(results) == limite:
                break
    return results


app = FastAPI()
app.mount("/api", API)
app.mount("/", StaticFiles(directory=DOCDIR), name="alexi")
middleware_args: dict[str, str | list[str]]
if os.getenv("DEVELOPMENT", False) or "dev" in sys.argv:
    LOGGER.info(
        "Running in development mode, will allow requests from http://localhost:*"
    )
    # Allow requests from localhost dev servers
    middleware_args = dict(
        allow_origin_regex="http://localhost(:.*)?",
    )
else:
    # Allow requests *only* from ZONALDA app (or otherwise configured site name)
    middleware_args = dict(
        allow_origins=[
            os.getenv("ORIGIN", "https://dhdaines.github.io"),
        ],
    )
app.add_middleware(CORSMiddleware, allow_methods=["GET", "OPTIONS"], **middleware_args)
