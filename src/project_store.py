"""Utilities for managing persisted RAG projects."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(slots=True)
class ProjectInfo:
    """Metadata describing a persisted project."""

    name: str
    display_name: str
    files: List[str]
    documents: int
    chunks: int

    @classmethod
    def empty(cls, name: str, display_name: str | None = None) -> "ProjectInfo":
        return cls(
            name=name,
            display_name=display_name or name,
            files=[],
            documents=0,
            chunks=0,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectInfo":
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            files=list(data.get("files", [])),
            documents=int(data.get("documents", 0)),
            chunks=int(data.get("chunks", 0)),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["files"] = list(self.files)
        return payload


class ProjectStore:
    """Manage project lifecycle and metadata on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def sanitize_name(self, raw: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
        cleaned = cleaned.strip("-_.").lower()
        if not cleaned:
            raise ValueError("Project name must include at least one alphanumeric character.")
        return cleaned

    def project_path(self, slug: str) -> Path:
        return self.root / slug

    def metadata_path(self, slug: str) -> Path:
        return self.project_path(slug) / "metadata.json"

    def vector_path(self, slug: str, *, ensure: bool = True) -> Path:
        path = self.project_path(slug) / "vector"
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def exists(self, name: str) -> bool:
        slug = self.sanitize_name(name)
        return self.project_path(slug).exists()

    def ensure_project(
        self,
        name: str,
        *,
        create: bool,
        reset: bool = False,
    ) -> ProjectInfo:
        display_name = name.strip() or name
        slug = self.sanitize_name(display_name)
        path = self.project_path(slug)
        if not path.exists():
            if not create:
                raise FileNotFoundError(f"Project '{display_name}' does not exist.")
            path.mkdir(parents=True, exist_ok=True)
            info = ProjectInfo.empty(slug, display_name or slug)
            self.save_info(info)
        else:
            info = self.load_info(slug)
            if not info.display_name:
                info.display_name = display_name or slug
        if reset:
            shutil.rmtree(self.vector_path(slug, ensure=False), ignore_errors=True)
            info = ProjectInfo.empty(slug, display_name or info.display_name or slug)
            self.save_info(info)
        return info

    def load_info(self, name: str) -> ProjectInfo:
        slug = self.sanitize_name(name)
        path = self.metadata_path(slug)
        if not path.exists():
            info = ProjectInfo.empty(slug)
            self.save_info(info)
            return info
        data = json.loads(path.read_text(encoding="utf-8"))
        info = ProjectInfo.from_dict(data)
        if not info.display_name:
            info.display_name = info.name
        return info

    def save_info(self, info: ProjectInfo) -> None:
        slug = self.sanitize_name(info.name)
        path = self.metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def list_infos(self) -> List[ProjectInfo]:
        infos: List[ProjectInfo] = []
        for entry in sorted(self.root.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                try:
                    infos.append(self.load_info(entry.name))
                except ValueError:
                    continue
        infos.sort(key=lambda info: info.display_name.lower())
        return infos

    def options(self) -> List[Tuple[str, str]]:
        return [(info.display_name, info.name) for info in self.list_infos()]


__all__ = ["ProjectStore", "ProjectInfo"]

