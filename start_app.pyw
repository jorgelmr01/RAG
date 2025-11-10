"""Self-contained launcher for the Document RAG Assistant."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

try:
    from tkinter import messagebox, Tk
except Exception:  # pragma: no cover - tkinter may be missing in rare installs
    messagebox = None  # type: ignore
    Tk = None  # type: ignore


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
SCRIPTS_DIR = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PY = SCRIPTS_DIR / ("python.exe" if os.name == "nt" else "python")
VENV_PYW = SCRIPTS_DIR / ("pythonw.exe" if os.name == "nt" else "python")
REQUIREMENTS = ROOT / "requirements.txt"
STAMP_FILE = VENV_DIR / "requirements.fingerprint"


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


def _show_dialog(title: str, body: str, *, error: bool = False) -> None:
    if messagebox is None or Tk is None:
        return
    try:
        root = Tk()
        root.withdraw()
        if error:
            messagebox.showerror(title, body)
        else:
            messagebox.showinfo(title, body)
        root.destroy()
    except Exception:
        pass


def _show_error(title: str, body: str) -> None:
    _show_dialog(title, body, error=True)


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


def _run_subprocess(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"Command failed: {' '.join(args)}\n\n{output.strip()}")


def _ensure_virtualenv() -> None:
    if VENV_PY.exists():
        return
    _show_dialog(
        "Document RAG Assistant",
        "Setting up the Python environment. This runs once and may take a minute.",
    )
    _run_subprocess([sys.executable, "-m", "venv", str(VENV_DIR)])


def _requirements_fingerprint() -> str | None:
    if not REQUIREMENTS.exists():
        return None
    stat = REQUIREMENTS.stat()
    return f"{stat.st_size}|{stat.st_mtime_ns}"


def _install_requirements(python_exe: Path) -> None:
    fingerprint = _requirements_fingerprint()
    if fingerprint is None:
        return
    if STAMP_FILE.exists():
        saved = STAMP_FILE.read_text(encoding="utf-8").strip()
        if saved == fingerprint:
            return
    _show_dialog(
        "Document RAG Assistant",
        "Installing project dependencies. This may take a minute—please wait.",
    )
    _run_subprocess([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    _run_subprocess([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(fingerprint, encoding="utf-8")


def _running_inside_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except FileNotFoundError:
        return False


def _bootstrap_environment() -> bool:
    """Ensure the virtual environment and dependencies exist.

    Returns True if this process should continue to launch the app, False if a
    child process has been spawned and the current process should exit.
    """

    _ensure_virtualenv()

    venv_python = VENV_PY
    if not venv_python.exists():
        raise RuntimeError("Virtual environment appears to be corrupted. Delete '.venv' and retry.")

    # Install dependencies using the venv's console Python
    console_python = venv_python
    _install_requirements(console_python)

    # If we're already inside the venv, continue in this process.
    if _running_inside_venv():
        return True

    # Relaunch this script inside the virtual environment using pythonw (GUI) when available.
    pythonw = VENV_PYW if VENV_PYW.exists() else venv_python
    subprocess.Popen([str(pythonw), str(ROOT / "start_app.pyw")], env=os.environ.copy())
    return False


def main() -> None:
    if not _bootstrap_environment():
        return

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

