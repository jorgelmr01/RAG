"""Console-based launcher for the Document RAG Assistant on Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
SCRIPTS_DIR = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PY = SCRIPTS_DIR / ("python.exe" if os.name == "nt" else "python")
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
    print("\nThis will set up everything automatically. Please wait...\n")


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"\n$ {' '.join(args)}")
    result = subprocess.run(
        args,
        cwd=str(cwd or ROOT),
        check=False,  # Don't raise on error, we'll handle it
        text=True,
        capture_output=capture_output,
    )
    if result.returncode != 0:
        # Re-raise with captured output for better error messages
        exc = subprocess.CalledProcessError(result.returncode, args)
        exc.stdout = result.stdout
        exc.stderr = result.stderr
        raise exc
    return result


def requirements_fingerprint() -> str | None:
    if not REQUIREMENTS.exists():
        return None
    stat = REQUIREMENTS.stat()
    return f"{stat.st_size}|{stat.st_mtime_ns}"


def modules_missing(python_exe: Path) -> list[str]:
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        try:
            run_command(
                [str(python_exe), "-c", f"import {module}"],
                capture_output=True,
                quiet=True,
            )
        except subprocess.CalledProcessError:
            missing.append(module)
    return missing


def ensure_virtualenv() -> None:
    if VENV_PY.exists():
        return
    print("\nCreating local virtual environment (.venv)…")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT)


def ensure_dependencies() -> None:
    fingerprint = requirements_fingerprint()
    missing = modules_missing(VENV_PY)
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
        "setuptools",
        "wheel",
    ]
    run_command(pip_args)

    # Install dependencies using only pre-built wheels to avoid compilation issues
    install_args = [
        str(VENV_PY),
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
        "--upgrade",
        "--only-binary", ":all:",  # Force use of pre-built wheels only
        "--prefer-binary",  # Prefer binary wheels when available
        "-r",
        str(REQUIREMENTS),
    ]
    try:
        # Capture output to detect compilation errors
        result = subprocess.run(
            install_args,
            cwd=str(ROOT),
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            # Print the output so user can see what went wrong
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            # Re-raise with captured output
            exc = subprocess.CalledProcessError(result.returncode, install_args)
            exc.stdout = result.stdout
            exc.stderr = result.stderr
            raise exc
    except subprocess.CalledProcessError:
        raise  # Re-raise to be caught by main error handler

    if fingerprint is not None:
        STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STAMP_FILE.write_text(fingerprint, encoding="utf-8")


def launch_app() -> None:
    python_for_app = VENV_PY
    print("\n" + "=" * 68)
    print(" Launching the assistant...".center(68))
    print("=" * 68)
    print("\n✅ Setup complete! The app will open in your browser shortly.")
    print("   (Press Ctrl+C to stop the app)\n")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.call([str(python_for_app), str(ROOT / "app.py")], cwd=str(ROOT))


def main() -> None:
    print_header()
    ensure_virtualenv()
    ensure_dependencies()
    launch_app()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        print("You can close this window.")
    except subprocess.CalledProcessError as exc:
        print(f"\n\n❌ Error: Command failed with exit code {exc.returncode}")
        print(f"Command: {' '.join(exc.cmd)}")
        
        # Check if it's a compilation error
        error_output = ""
        if hasattr(exc, 'stderr') and exc.stderr:
            error_output = exc.stderr.lower()
        elif hasattr(exc, 'stdout') and exc.stdout:
            error_output = exc.stdout.lower()
        
        is_compilation_error = any(
            keyword in error_output
            for keyword in [
                "mesonpy",
                "numpy",
                "gcc",
                "cl.exe",
                "compiler",
                "metadata-generation-failed",
                "cannot find the file specified",
                "el sistema no puede encontrar",
            ]
        )
        
        print("\nTroubleshooting:")
        if is_compilation_error:
            print("⚠️  This looks like a compilation error (trying to build from source).")
            print("   This usually happens when pre-built packages aren't available.")
            print("\n   Solutions:")
            print("   1. Make sure you have Python 3.11 or 3.12 (not 3.13+)")
            print("   2. Try deleting the .venv folder and running again")
            print("   3. Make sure you have internet connection")
            print("   4. Try updating pip: python -m pip install --upgrade pip")
            print("\n   If the problem persists:")
            print("   - Delete the .venv folder completely")
            print("   - Make sure you're using Python 3.11 or 3.12")
            print("   - Run the launcher again")
        else:
            print("1. Make sure Python 3.11+ is installed")
            print("2. Check that you have internet connection")
            print("3. Try deleting the .venv folder and running again")
            print("4. Try running again - sometimes it's a temporary issue")
        
        input("\nPress Enter to close...")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\n\n❌ Unexpected error: {exc}")
        print("\nTroubleshooting:")
        print("1. Make sure Python 3.11+ is installed")
        print("2. Check the error message above")
        print("3. Try deleting the .venv folder and running again")
        input("\nPress Enter to close...")
        sys.exit(1)

