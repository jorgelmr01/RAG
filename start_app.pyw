"""Double-click entry point for the Document RAG Assistant on Windows."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

try:
    from tkinter import messagebox, Tk
except Exception:  # pragma: no cover - tkinter may be missing in rare installs
    messagebox = None  # type: ignore
    Tk = None  # type: ignore


ROOT = Path(__file__).resolve().parent


class _NullStream:
    """Fallback stream implementing the minimal interface uvicorn expects."""

    def __init__(self, name: str) -> None:
        self.name = name

    def write(self, *_: object) -> None:  # pragma: no cover - no-op
        pass

    def flush(self) -> None:  # pragma: no cover - no-op
        pass

    def isatty(self) -> bool:
        return False


def _show_error(title: str, body: str) -> None:
    if messagebox is None or Tk is None:
        return
    try:
        root = Tk()
        root.withdraw()
        messagebox.showerror(title, body)
        root.destroy()
    except Exception:
        pass


def _write_log(exc: BaseException) -> Path:
    log_path = ROOT / "last_error.log"
    formatted = "\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_path.write_text(formatted, encoding="utf-8")
    return log_path


def _prepare_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if sys.stdout is None:
        sys.stdout = _NullStream("stdout")  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullStream("stderr")  # type: ignore[assignment]


def main() -> None:
    _prepare_path()
    try:
        from app import launch_app  # local import after path adjustment
    except Exception as exc:  # pragma: no cover - surfaced on runtime import issue
        log_path = _write_log(exc)
        _show_error(
            "Document RAG Assistant",
            (
                "Failed to start the application.\n\n"
                f"Details were written to: {log_path}\n"
                "If dependencies are missing, run 'pip install -r requirements.txt'."
            ),
        )
        raise

    try:
        launch_app()
    except Exception as exc:  # pragma: no cover - runtime errors are logged
        log_path = _write_log(exc)
        _show_error(
            "Document RAG Assistant",
            (
                "The app encountered an unexpected error and closed.\n\n"
                f"Details were written to: {log_path}\n"
                "Please share the log if you need support."
            ),
        )
        raise


if __name__ == "__main__":
    main()

