"""Core retrieval augmented generation pipeline."""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig
from .document_loaders import load_documents
from .project_store import ProjectInfo, ProjectStore


class DocumentIngestionError(RuntimeError):
    """Raised when the ingestion pipeline cannot process the supplied files."""


def _clean_text(value: str, *, limit: int) -> str:
    """Return a whitespace-normalised snippet capped to ``limit`` characters."""

    stripped = re.sub(r"\s+", " ", value).strip()
    if limit <= 0 or len(stripped) <= limit:
        return stripped
    return textwrap.shorten(stripped, width=limit, placeholder=" …")


def _format_history(history: Sequence[tuple[str, str]], max_turns: int) -> str:
    if not history:
        return "(no prior conversation)"
    recent = history[-max_turns:]
    return "\n".join(
        f"User: {human}\nAssistant: {assistant}" for human, assistant in recent
    )


def _build_messages(
    question: str,
    context_block: str,
    history_block: str,
) -> List[BaseMessage]:
    system_prompt = (
        "You are a meticulous research assistant. "
        "Answer the user's question strictly using the provided context. "
        "Cite the supporting snippets with bracketed references like [1] tied to the context order. "
        "If the context lacks the answer, say you do not have enough information."
    )
    user_prompt = (
        f"Conversation so far:\n{history_block}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Formulate a helpful, concise answer in markdown."
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


@dataclass
class KnowledgeStats:
    files: set[str] = field(default_factory=set)
    documents: int = 0
    chunks: int = 0


class RAGPipeline:
    """High-level API managing ingestion, retrieval and generation."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.project_store = ProjectStore(Path(self.config.projects_path))
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._llm: Optional[ChatOpenAI] = None
        self._api_key: Optional[str] = None
        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        self.stats = KnowledgeStats()
        self._last_documents: List[Document] = []
        self.current_project: Optional[ProjectInfo] = None

    # ------------------------------------------------------------------
    # Client initialisation helpers
    # ------------------------------------------------------------------
    def configure_api_key(self, api_key: Optional[str] = None) -> None:
        """Set (or refresh) the API credentials used by the pipeline."""

        key = (api_key or self._api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "No OpenAI API key found. Provide one via the UI or set OPENAI_API_KEY."
            )
        
        # Validate key format
        if not key.startswith("sk-"):
            raise ValueError(
                "API key should start with 'sk-'. Please check that you copied the complete key."
            )
        
        if len(key) < 20:
            raise ValueError(
                "API key appears to be truncated. OpenAI keys are typically 51 characters long. "
                "Please copy the complete key from https://platform.openai.com/account/api-keys"
            )
        
        # Store the key for future use
        self._api_key = key
        os.environ["OPENAI_API_KEY"] = key

        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=key,
            )
            self._llm = ChatOpenAI(
                model=self.config.chat_model,
                temperature=self.config.temperature,
                streaming=True,
                api_key=key,
            )
        except Exception as exc:
            error_str = str(exc).lower()
            if "invalid_api_key" in error_str or "incorrect api key" in error_str or "401" in error_str:
                raise ValueError(
                    f"Invalid API key. The key you provided was rejected by OpenAI. "
                    f"Please verify it at https://platform.openai.com/account/api-keys"
                ) from exc
            raise
        
        # If we have a current project but no vector store yet, initialize it now
        if self.current_project is not None and self.vector_store is None:
            self._initialise_vector_store(self.current_project)

    def _ensure_clients(self) -> None:
        """Ensure OpenAI clients are initialized. Only initializes if API key is available."""
        if self._embeddings is None or self._llm is None:
            # Try to configure API key, but don't raise error if not available
            # This allows the pipeline to be initialized without an API key
            try:
                self.configure_api_key()
            except ValueError:
                # No API key available yet - this is OK, we'll require it when needed
                pass

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    @property
    def has_project(self) -> bool:
        return self.current_project is not None

    @property
    def has_knowledge(self) -> bool:
        return self.vector_store is not None and self.stats.chunks > 0

    def list_projects(self) -> List[ProjectInfo]:
        return self.project_store.list_infos()

    def project_options(self) -> List[Tuple[str, str]]:
        return self.project_store.options()

    def project_status(self) -> str:
        if not self.current_project:
            return "No project selected."
        file_count = len(self.stats.files)
        return (
            f"**Current project:** `{self.current_project.display_name}`\n"
            f"{self.stats.chunks} chunks across {file_count} source files."
        )

    def ensure_project_selected(self) -> ProjectInfo:
        if self.current_project:
            return self.current_project
        existing = self.project_store.list_infos()
        if existing:
            return self._initialise_vector_store(existing[0])
        return self.create_project("default")

    def create_project(self, name: str) -> ProjectInfo:
        info = self.project_store.ensure_project(name, create=True, reset=True)
        return self._initialise_vector_store(info)

    def load_project(self, name: str) -> ProjectInfo:
        slug = self.project_store.sanitize_name(name)
        if not self.project_store.exists(slug):
            raise ValueError(f"Project '{name}' does not exist yet.")
        info = self.project_store.load_info(slug)
        return self._initialise_vector_store(info)

    def reset_current_project(self) -> KnowledgeStats:
        info = self.ensure_project_selected()
        refreshed = self.project_store.ensure_project(info.display_name, create=True, reset=True)
        self._initialise_vector_store(refreshed)
        self._save_project_info()
        return self.stats

    def _initialise_vector_store(self, info: ProjectInfo) -> ProjectInfo:
        """Initialize the vector store for a project. Requires API key to be configured."""
        self._ensure_clients()
        
        # If we don't have embeddings yet (no API key), we can't initialize the vector store
        # Just set the project info and stats, but leave vector_store as None
        if self._embeddings is None:
            self.stats = KnowledgeStats(
                files=set(info.files),
                documents=info.documents,
                chunks=info.chunks,
            )
            self._last_documents = []
            self.current_project = info
            self.vector_store = None
            self.retriever = None
            return info
        
        # We have embeddings, so we can initialize the vector store
        vector_dir = self.project_store.vector_path(info.name, ensure=True)
        self.vector_store = Chroma(
            collection_name=f"project-{info.name}",
            embedding_function=self._embeddings,
            persist_directory=str(vector_dir),
        )
        search_kwargs = {"k": self.config.top_k}
        if self.config.score_threshold > 0:
            search_kwargs["score_threshold"] = self.config.score_threshold
            search_type = "similarity_score_threshold"
        else:
            search_type = "similarity"
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        self.stats = KnowledgeStats(
            files=set(info.files),
            documents=info.documents,
            chunks=info.chunks,
        )
        self._last_documents = []
        self.current_project = info
        return info

    def _save_project_info(self) -> None:
        if not self.current_project:
            return
        info = ProjectInfo(
            name=self.current_project.name,
            display_name=self.current_project.display_name,
            files=sorted(self.stats.files),
            documents=self.stats.documents,
            chunks=self.stats.chunks,
        )
        self.project_store.save_info(info)
        self.current_project = info

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, file_paths: Iterable[str | Path], *, append: bool = False) -> KnowledgeStats:
        """Load, split and embed the provided files."""

        project = self.ensure_project_selected()
        self._ensure_clients()
        if self._embeddings is None:
            raise ValueError(
                "OpenAI API key is required to ingest documents. "
                "Please configure your API key in the Configuration section."
            )
        documents = load_documents(file_paths)
        if not documents:
            raise DocumentIngestionError(
                "No readable documents were found in the uploaded files."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise DocumentIngestionError(
                "Document splitting produced zero chunks. Adjust chunk settings or verify contents."
            )

        assert self.vector_store is not None  # ensured by ensure_project_selected
        ids = [str(uuid4()) for _ in chunks]
        self.vector_store.add_documents(documents=chunks, ids=ids)
        # Note: ChromaDB with persist_directory automatically persists data,
        # so no explicit persist() call is needed

        self.stats.files.update({doc.metadata.get("source", "") for doc in documents})
        self.stats.documents += len(documents)
        self.stats.chunks += len(chunks)
        self._save_project_info()
        # Ensure retriever reflects updated config
        self._initialise_vector_store(self.current_project)
        return self.stats

    # ------------------------------------------------------------------
    # Retrieval and generation
    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> List[Document]:
        self.ensure_project_selected()
        self._ensure_clients()
        if self._embeddings is None:
            raise ValueError(
                "OpenAI API key is required to retrieve documents. "
                "Please configure your API key in the Configuration section."
            )
        if not self.has_knowledge or self.retriever is None:
            raise RuntimeError("No documents indexed. Upload files before asking a question.")
        assert self.vector_store is not None  # safety
        results: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(
            question,
            k=max(self.config.top_k, self.config.max_context_sections),
        )

        filtered: List[Document] = []
        for document, score in results:
            if self.config.score_threshold > 0 and score > self.config.score_threshold:
                continue
            filtered.append(document)
        if not filtered and results:
            filtered = [doc for doc, _ in results]

        limited = filtered[: self.config.max_context_sections]
        if not limited:
            raise RuntimeError(
                "No relevant context found for this question. Try rephrasing or upload more documents."
            )

        self._last_documents = limited
        return self._last_documents

    def stream_answer(
        self,
        question: str,
        docs: Sequence[Document],
        chat_history: Sequence[tuple[str, str]],
    ) -> Generator[str, None, None]:
        self._ensure_clients()
        if self._llm is None:
            raise RuntimeError("Language model not initialised.")

        context = self._format_context(docs)
        history_block = _format_history(chat_history, self.config.history_turns)
        messages = _build_messages(question, context, history_block)

        response = ""
        for chunk in self._llm.stream(messages):
            if isinstance(chunk, AIMessageChunk):
                response += chunk.content
                yield chunk.content
            else:  # pragma: no cover - defensive branch
                content = getattr(chunk, "content", "")
                if content:
                    response += content
                    yield content

    # ------------------------------------------------------------------
    # Formatting helpers for the UI
    # ------------------------------------------------------------------
    def _format_context(self, docs: Sequence[Document]) -> str:
        if not docs:
            return "(no supporting context retrieved)"
        snippets: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            raw_source = doc.metadata.get("source") or self.config.fallback_source_label
            try:
                source = Path(str(raw_source)).name or self.config.fallback_source_label
            except Exception:
                source = str(raw_source)
            page = doc.metadata.get("page")
            if isinstance(page, int):
                source = f"{source} (page {page + 1})"
            elif isinstance(page, str) and page.isdigit():
                source = f"{source} (page {int(page) + 1})"
            snippet = _clean_text(doc.page_content, limit=self.config.max_context_chars)
            snippets.append(f"[{idx}] {source}\n{snippet}")
        return "\n\n".join(snippets)

    def render_loaded_sources(self) -> str:
        if not self.stats.files:
            return "No documents indexed yet."
        lines = []
        for name in sorted(self.stats.files):
            if not name:
                continue
            display = Path(name).name if name else self.config.fallback_source_label
            lines.append(f"- `{display}`")
        return "\n".join(lines)

    def render_sources(self, docs: Optional[Sequence[Document]] = None) -> str:
        docs = list(docs or self._last_documents)
        if not docs:
            return "No supporting context retrieved for the last answer."
        lines = []
        for idx, doc in enumerate(docs, start=1):
            raw_source = doc.metadata.get("source") or self.config.fallback_source_label
            try:
                source = Path(str(raw_source)).name or self.config.fallback_source_label
            except Exception:
                source = str(raw_source)
            page = doc.metadata.get("page")
            location = ""
            if isinstance(page, int):
                location = f" (page {page + 1})"
            elif isinstance(page, str) and page.isdigit():
                location = f" (page {int(page) + 1})"
            lines.append(f"[{idx}] `{source}`{location}")
        return "\n".join(f"- {line}" for line in lines)


__all__ = ["RAGPipeline", "DocumentIngestionError", "KnowledgeStats"]

