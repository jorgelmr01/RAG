"""Self-contained launcher for the Document RAG Assistant."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover - tkinter may be missing in rare installs
    tk = None  # type: ignore
    messagebox = None  # type: ignore
    ScrolledText = None  # type: ignore


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


class _StatusWindow:
    """Lightweight status window for long-running setup steps."""

    def __init__(self, title: str) -> None:
        self._enabled = tk is not None and ScrolledText is not None
        self._root = None
        self._header = None
        self._log = None
        if not self._enabled:
            print(f"[{title}]")
            return
        try:
            root = tk.Tk()
            root.title(title)
            root.geometry("520x260")
            root.resizable(False, False)
            root.attributes("-topmost", True)
            header = tk.Label(root, text="Preparing…", anchor="w", font=("Segoe UI", 11, "bold"))
            header.pack(fill="x", padx=20, pady=(18, 6))
            log = ScrolledText(root, height=9, width=60, state="disabled", font=("Consolas", 9))
            log.pack(fill="both", expand=True, padx=20, pady=(0, 18))
            root.update()
            self._root = root
            self._header = header
            self._log = log
        except Exception:
            self._enabled = False
            self._root = None
            self._header = None
            self._log = None

    def pump(self) -> None:
        if self._enabled and self._root is not None:
            self._root.update_idletasks()
            self._root.update()

    def step(self, message: str) -> None:
        if self._enabled and self._header is not None:
            self._header.configure(text=message)
            self.pump()
        else:
            print(message)

    def append(self, line: str) -> None:
        if self._enabled and self._log is not None:
            self._log.configure(state="normal")
            self._log.insert("end", line + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
            self.pump()
        else:
            print(line)

    def close(self) -> None:
        if self._enabled and self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._header = None
        self._log = None


def _show_dialog(title: str, body: str, *, error: bool = False) -> None:
    if messagebox is None or tk is None:
        return
    try:
        root = tk.Tk()
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


def _run_command(args: list[str], status: _StatusWindow, description: str | None = None) -> None:
    if description:
        status.step(description)
    status.append(f"$ {' '.join(args)}")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(ROOT),
        creationflags=creationflags,
    )
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            status.pump()
            continue
        status.append(line.rstrip())
    process.stdout.close()
    process.wait()
    status.pump()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(args)}")


def _ensure_virtualenv(status: _StatusWindow) -> None:
    if VENV_PY.exists():
        return
    _run_command([sys.executable, "-m", "venv", str(VENV_DIR)], status, "Creating Python virtual environment…")


def _requirements_fingerprint() -> str | None:
    if not REQUIREMENTS.exists():
        return None
    stat = REQUIREMENTS.stat()
    return f"{stat.st_size}|{stat.st_mtime_ns}"


def _dependencies_present(python_exe: Path) -> bool:
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.run(
            [
                str(python_exe),
                "-c",
                "import gradio,langchain,langchain_openai,langchain_chroma",
            ],
            check=True,
            cwd=str(ROOT),
            creationflags=creationflags,
        )
        return True
    except Exception:
        return False


def _install_requirements(python_exe: Path, status: _StatusWindow) -> None:
    fingerprint = _requirements_fingerprint()
    if fingerprint is None:
        return
    need_install = True
    if STAMP_FILE.exists():
        saved = STAMP_FILE.read_text(encoding="utf-8").strip()
        if saved == fingerprint and _dependencies_present(python_exe):
            need_install = False
    if not need_install:
        return
    _run_command(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        status,
        "Upgrading pip…",
    )
    _run_command(
        [str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        status,
        "Installing project dependencies… (this may take a minute)",
    )
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(fingerprint, encoding="utf-8")


def _running_inside_venv() -> bool:
    try:
        return Path(sys.prefix).resolve() == VENV_DIR.resolve()
    except FileNotFoundError:
        return False


def _bootstrap_environment(status: _StatusWindow) -> bool:
    """Ensure the virtual environment and dependencies exist.

    Returns True if this process should continue to launch the app, False if a
    child process has been spawned and the current process should exit.
    """

    status.step("Checking Python environment…")
    _ensure_virtualenv(status)

    venv_python = VENV_PY
    if not venv_python.exists():
        raise RuntimeError("Virtual environment appears to be corrupted. Delete '.venv' and retry.")

    # Install dependencies using the venv's console Python
    console_python = venv_python
    _install_requirements(console_python, status)

    # If we're already inside the venv, continue in this process.
    if _running_inside_venv():
        return True

    # Relaunch this script inside the virtual environment using pythonw (GUI) when available.
    status.step("Launching assistant…")
    pythonw = VENV_PYW if VENV_PYW.exists() else venv_python
    subprocess.Popen([str(pythonw), str(ROOT / "start_app.pyw")], env=os.environ.copy())
    return False


def main() -> None:
    status = _StatusWindow("Document RAG Assistant")
    try:
        if not _bootstrap_environment(status):
            status.close()
            return
        status.step("Starting app…")

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
            status.close()
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
    finally:
        status.close()


if __name__ == "__main__":
    main()

