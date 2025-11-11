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

# Optional loaders - imported conditionally to avoid breaking if dependencies are missing
try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    HAS_PPT = True
except ImportError:
    HAS_PPT = False

try:
    from langchain_community.document_loaders import BSHTMLLoader
    HAS_HTML = True
except ImportError:
    HAS_HTML = False

try:
    from langchain_community.document_loaders import JSONLoader
    HAS_JSON = True
except ImportError:
    HAS_JSON = False


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


def _excel_loader(path: Path) -> Sequence[Document]:
    """Load Excel files (.xlsx, .xls)."""
    if not HAS_EXCEL:
        raise RuntimeError(
            "Excel support requires additional dependencies. "
            "Install with: pip install openpyxl unstructured[excel]"
        )
    return UnstructuredExcelLoader(str(path)).load()


def _pptx_loader(path: Path) -> Sequence[Document]:
    """Load PowerPoint files (.pptx, .ppt)."""
    if not HAS_PPT:
        raise RuntimeError(
            "PowerPoint support requires additional dependencies. "
            "Install with: pip install python-pptx unstructured[ppt]"
        )
    return UnstructuredPowerPointLoader(str(path)).load()


def _html_loader(path: Path) -> Sequence[Document]:
    """Load HTML files."""
    if not HAS_HTML:
        # Fallback to text loader for HTML
        return _text_loader(path)
    return BSHTMLLoader(str(path)).load()


def _json_loader(path: Path) -> Sequence[Document]:
    """Load JSON files."""
    if not HAS_JSON:
        # Fallback to text loader for JSON
        return _text_loader(path)
    # JSONLoader requires a jq_schema - use simple text extraction for now
    return _text_loader(path)


SUPPORTED_EXTENSIONS: dict[str, LoaderFactory] = {
    ".pdf": _pdf_loader,
    ".txt": _text_loader,
    ".md": _text_loader,
    ".rtf": _text_loader,
    ".csv": partial(_csv_loader, delimiter=","),
    ".tsv": partial(_csv_loader, delimiter="\t"),
    ".docx": _docx_loader,
    # Optional loaders - will fall back to text if dependencies missing
    ".xlsx": _excel_loader if HAS_EXCEL else _text_loader,
    ".xls": _excel_loader if HAS_EXCEL else _text_loader,
    ".pptx": _pptx_loader if HAS_PPT else _text_loader,
    ".ppt": _pptx_loader if HAS_PPT else _text_loader,
    ".html": _html_loader,
    ".htm": _html_loader,
    ".json": _json_loader,
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

