"""Utilities for loading heterogeneous document types into LangChain."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)


LOGGER = logging.getLogger(__name__)


LoaderFactory = Callable[[Path], Sequence[Document]]


def _pdf_loader(path: Path) -> Sequence[Document]:
    return PyPDFLoader(str(path)).load()


def _text_loader(path: Path) -> Sequence[Document]:
    return TextLoader(str(path), autodetect_encoding=True).load()


def _csv_loader(path: Path, *, delimiter: str = ",") -> Sequence[Document]:
    return CSVLoader(
        str(path),
        encoding="utf-8",
        csv_args={"delimiter": delimiter},
    ).load()


def _docx_loader(path: Path) -> Sequence[Document]:
    return Docx2txtLoader(str(path)).load()


SUPPORTED_EXTENSIONS: dict[str, LoaderFactory] = {
    ".pdf": _pdf_loader,
    ".txt": _text_loader,
    ".md": _text_loader,
    ".rtf": _text_loader,
    ".csv": partial(_csv_loader, delimiter=","),
    ".tsv": partial(_csv_loader, delimiter="\t"),
    ".docx": _docx_loader,
}


def _normalise_metadata(documents: Sequence[Document], file_path: Path) -> List[Document]:
    """Ensure each document carries consistent metadata."""

    normalised: List[Document] = []
    for doc in documents:
        metadata = dict(doc.metadata)
        metadata.setdefault("source", file_path.name)
        metadata.setdefault("source_path", str(file_path))
        metadata.setdefault("file_type", file_path.suffix.lower())
        doc.metadata = metadata
        normalised.append(doc)
    return normalised


def load_file(path: Path) -> List[Document]:
    """Load a single file into a list of LangChain documents."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    loader = SUPPORTED_EXTENSIONS.get(path.suffix.lower(), _text_loader)
    try:
        documents = loader(path)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(f"Unable to read '{path.name}': {exc}") from exc

    return _normalise_metadata(documents, path)


def load_documents(file_paths: Iterable[str | Path]) -> List[Document]:
    """Load all provided files and return a single flat list of documents."""

    documents: List[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        try:
            documents.extend(load_file(path))
        except Exception as exc:
            LOGGER.warning("Skipping '%s': %s", path, exc)
    return documents


__all__ = ["load_documents", "load_file", "SUPPORTED_EXTENSIONS"]

