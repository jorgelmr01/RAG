"""Console-based launcher for the Document RAG Assistant on Windows."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
SCRIPTS_DIR = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
VENV_PY = SCRIPTS_DIR / ("python.exe" if os.name == "nt" else "python")
REQUIREMENTS = ROOT / "requirements.txt"
STAMP_FILE = VENV_DIR / "requirements.fingerprint"
ERROR_LOG = ROOT / "error_log.txt"
REQUIRED_MODULES = [
    "gradio",
    "langchain",
    "langchain_openai",
    "langchain_chroma",
    "langchain_community",
]


def get_system_info() -> str:
    """Collect system information for error logs."""
    info = []
    info.append(f"Platform: {platform.platform()}")
    info.append(f"System: {platform.system()} {platform.release()}")
    info.append(f"Architecture: {platform.machine()}")
    info.append(f"Python Version: {sys.version}")
    info.append(f"Python Executable: {sys.executable}")
    info.append(f"Python Path: {sys.path}")
    info.append(f"Working Directory: {os.getcwd()}")
    info.append(f"Script Directory: {ROOT}")
    info.append(f"Virtual Environment: {VENV_DIR} (exists: {VENV_DIR.exists()})")
    if VENV_PY.exists():
        try:
            result = subprocess.run(
                [str(VENV_PY), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            info.append(f"VENV Python: {result.stdout.strip()}")
        except Exception:
            info.append(f"VENV Python: Could not determine version")
    return "\n".join(info)


def write_error_log(error_type: str, error: Exception, context: dict | None = None) -> None:
    """Write detailed error information to a log file."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ERROR LOG - {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(get_system_info())
            f.write("\n\n")
            
            f.write(f"ERROR TYPE: {error_type}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exception: {type(error).__name__}\n")
            f.write(f"Message: {str(error)}\n\n")
            
            if context:
                f.write("CONTEXT:\n")
                f.write("-" * 80 + "\n")
                for key, value in context.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("TRACEBACK:\n")
            f.write("-" * 80 + "\n")
            f.write("".join(traceback.format_exception(type(error), error, error.__traceback__)))
            f.write("\n\n")
            
            if isinstance(error, subprocess.CalledProcessError):
                f.write("COMMAND OUTPUT:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Command: {' '.join(error.cmd)}\n")
                f.write(f"Return Code: {error.returncode}\n")
                if hasattr(error, 'stdout') and error.stdout:
                    f.write(f"\nSTDOUT:\n{error.stdout}\n")
                if hasattr(error, 'stderr') and error.stderr:
                    f.write(f"\nSTDERR:\n{error.stderr}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n\n")
    except Exception as log_error:
        # If we can't write the log, at least print it
        print(f"\n⚠️  Could not write error log: {log_error}")


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

    # Try installation with multiple strategies to handle different scenarios
    install_strategies = [
        # Strategy 1: Prefer binary wheels (best for most users, avoids compilation)
        {
            "name": "prefer-binary",
            "args": [
                str(VENV_PY),
                "-m",
                "pip",
                "install",
                "--no-input",
                "--disable-pip-version-check",
                "--upgrade",
                "--prefer-binary",  # Prefer wheels but allow source if needed
                "-r",
                str(REQUIREMENTS),
            ],
        },
        # Strategy 2: No binary restrictions (fallback for dependency conflicts)
        {
            "name": "standard",
            "args": [
                str(VENV_PY),
                "-m",
                "pip",
                "install",
                "--no-input",
                "--disable-pip-version-check",
                "--upgrade",
                "-r",
                str(REQUIREMENTS),
            ],
        },
    ]
    
    last_error = None
    for strategy in install_strategies:
        try:
            print(f"\nTrying installation strategy: {strategy['name']}...")
            result = subprocess.run(
                strategy["args"],
                cwd=str(ROOT),
                check=False,
                text=True,
                capture_output=True,
            )
            if result.returncode == 0:
                print("✅ Installation successful!")
                break
            
            # Check if it's a dependency conflict (not a compilation error)
            output = (result.stdout + result.stderr).lower()
            is_dependency_conflict = any(
                keyword in output
                for keyword in [
                    "resolutionimpossible",
                    "conflicting dependencies",
                    "cannot install",
                    "conflicts have no matching distributions",
                ]
            )
            
            if is_dependency_conflict and strategy["name"] == "prefer-binary":
                # Dependency conflict with prefer-binary, try standard install
                print("⚠️  Dependency conflict detected, trying alternative approach...")
                last_error = result
                continue
            
            # For other errors or if we've tried all strategies, raise
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            exc = subprocess.CalledProcessError(result.returncode, strategy["args"])
            exc.stdout = result.stdout
            exc.stderr = result.stderr
            raise exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if strategy == install_strategies[-1]:
                # Last strategy failed, re-raise
                raise
            continue
    
    if last_error and last_error.returncode != 0:
        # If we get here, all strategies failed
        raise last_error

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
        sys.exit(0)
    except subprocess.CalledProcessError as exc:
        print(f"\n\n❌ Error: Command failed with exit code {exc.returncode}")
        print(f"Command: {' '.join(exc.cmd)}")
        
        # Log the error
        error_output = ""
        if hasattr(exc, 'stderr') and exc.stderr:
            error_output = exc.stderr.lower()
        if hasattr(exc, 'stdout') and exc.stdout:
            error_output += " " + exc.stdout.lower()
        
        context = {
            "Command": " ".join(exc.cmd),
            "Return Code": exc.returncode,
            "Working Directory": str(ROOT),
        }
        write_error_log("Command Execution Error", exc, context)
        
        is_compilation_error = any(
            keyword in error_output
            for keyword in [
                "mesonpy",
                "gcc",
                "cl.exe",
                "compiler",
                "metadata-generation-failed",
                "cannot find the file specified",
                "el sistema no puede encontrar",
                "building wheel",
                "building from source",
            ]
        )
        
        is_dependency_conflict = any(
            keyword in error_output
            for keyword in [
                "resolutionimpossible",
                "conflicting dependencies",
                "conflicts have no matching distributions",
                "cannot install",
            ]
        )
        
        print("\nTroubleshooting:")
        if is_dependency_conflict:
            print("⚠️  This is a dependency conflict (packages have incompatible requirements).")
            print("\n   Solutions:")
            print("   1. Make sure you have Python 3.11 or 3.12 (not 3.13+)")
            print("   2. Delete the .venv folder and try again")
            print("   3. Make sure you have internet connection")
            print("   4. Try updating pip: python -m pip install --upgrade pip")
            print("\n   If the problem persists:")
            print("   - Delete the .venv folder completely")
            print("   - Make sure you're using Python 3.11 or 3.12")
            print("   - Check that your Python installation is not corrupted")
            print("   - Try reinstalling Python from python.org")
        elif is_compilation_error:
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
        
        print(f"\n📝 Error details have been saved to: {ERROR_LOG}")
        print("   Please share this file if you need help troubleshooting.")
        input("\nPress Enter to close...")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\n\n❌ Unexpected error: {exc}")
        print("\nTroubleshooting:")
        print("1. Make sure Python 3.11+ is installed")
        print("2. Check the error message above")
        print("3. Try deleting the .venv folder and running again")
        
        # Log the unexpected error
        write_error_log("Unexpected Error", exc, {"Phase": "Main execution"})
        print(f"\n📝 Error details have been saved to: {ERROR_LOG}")
        print("   Please share this file if you need help troubleshooting.")
        input("\nPress Enter to close...")
        sys.exit(1)

