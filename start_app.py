"""Console-based launcher for the Document RAG Assistant on Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
SCRIPTS_DIR = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PY = SCRIPTS_DIR / ("python.exe" if os.name == "nt" else "python")
VENV_PYW = SCRIPTS_DIR / ("pythonw.exe" if os.name == "nt" else "python")
REQUIREMENTS = ROOT / "requirements.txt"
STAMP_FILE = VENV_DIR / "requirements.fingerprint"
REQUIRED_MODULES = [
    "gradio",
    "langchain",
    "langchain_openai",
    "langchain_chroma",
    "langchain_community",
]


def print_header() -> None:
    print("=" * 68)
    print(" Document RAG Assistant launcher".center(68))
    print("=" * 68)


def run_command(args: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.check_call(args, cwd=str(cwd or ROOT))


def requirements_fingerprint() -> str | None:
    if not REQUIREMENTS.exists():
        return None
    stat = REQUIREMENTS.stat()
    return f"{stat.st_size}|{stat.st_mtime_ns}"


def modules_missing() -> list[str]:
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        if find_spec(module) is None:
            missing.append(module)
    return missing


def ensure_virtualenv() -> None:
    if VENV_PY.exists():
        return
    print("\nCreating local virtual environment (.venv)…")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT)


def ensure_dependencies() -> None:
    fingerprint = requirements_fingerprint()
    missing = modules_missing()
    if not missing and fingerprint is not None and STAMP_FILE.exists():
        saved = STAMP_FILE.read_text(encoding="utf-8").strip()
        if saved == fingerprint:
            print("\nDependencies already satisfied.")
            return

    print("\nInstalling project dependencies (this happens only when needed)…")
    pip_args = [
        str(VENV_PY),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
    ]
    run_command(pip_args)

    install_args = [
        str(VENV_PY),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
        "-r",
        str(REQUIREMENTS),
    ]
    run_command(install_args)

    if fingerprint is not None:
        STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STAMP_FILE.write_text(fingerprint, encoding="utf-8")


def launch_app() -> None:
    python_for_app = VENV_PYW if VENV_PYW.exists() else VENV_PY
    print("\nLaunching the assistant… (press Ctrl+C to stop)")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.execv(str(python_for_app), [str(python_for_app), str(ROOT / "app.py")])


def main() -> None:
    print_header()
    ensure_virtualenv()

    # Re-run inside the virtual environment if needed.
    if Path(sys.prefix).resolve() != VENV_DIR.resolve():
        print("\nSwitching to the project virtual environment…")
        os.execv(
            str(VENV_PY),
            [
                str(VENV_PY),
                str(ROOT / "start_app.py"),
            ],
        )

    ensure_dependencies()
    launch_app()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        sys.exit(exc.returncode)

