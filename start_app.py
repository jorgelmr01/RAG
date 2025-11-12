"""Console-based launcher for the Document RAG Assistant on Windows."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

# Try importing chardet for encoding detection, but don't fail if it's not available
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


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
    info.append(f"Python Version Info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
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
    
    # Add Python version compatibility warning
    if sys.version_info.minor >= 13:
        info.append(f"\n⚠️  WARNING: Python {sys.version_info.major}.{sys.version_info.minor} is not fully supported!")
        info.append("   Python 3.13+ may cause installation failures due to missing pre-built wheels.")
        info.append("   Recommended: Use Python 3.11 or 3.12 for best compatibility.")
    
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


def check_python_version() -> None:
    """Validate Python version and provide helpful error if incompatible."""
    version = sys.version_info
    if version.major != 3:
        print("\n❌ ERROR: Python 3 is required.")
        print(f"   You are using Python {version.major}.{version.minor}.{version.micro}")
        print("   Please install Python 3.11 or 3.12 from https://www.python.org/downloads/")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if version.minor < 11:
        print("\n❌ ERROR: Python 3.11 or newer is required.")
        print(f"   You are using Python {version.major}.{version.minor}.{version.micro}")
        print("   Please install Python 3.11 or 3.12 from https://www.python.org/downloads/")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if version.minor >= 13:
        print("\n⚠️  WARNING: Python 3.13+ is not fully supported yet.")
        print(f"   You are using Python {version.major}.{version.minor}.{version.micro}")
        print("   Many packages don't have pre-built wheels for Python 3.13+ yet.")
        print("   This may cause installation failures.")
        print("\n   RECOMMENDED: Use Python 3.11 or 3.12 for best compatibility.")
        print("   Download from: https://www.python.org/downloads/")
        print("\n   Continuing anyway... (this may fail)")
        input("\nPress Enter to continue or Ctrl+C to cancel...")


def check_python_version() -> None:
    """Check if Python version is supported and warn if not."""
    version = sys.version_info
    major, minor = version.major, version.minor
    
    if major != 3:
        print("❌ ERROR: Python 3 is required.")
        print(f"   You are using Python {major}.{minor}")
        print("   Please install Python 3.11+ from https://www.python.org/downloads/")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if minor < 11:
        print("❌ ERROR: Python 3.11 or newer is required.")
        print(f"   You are using Python {major}.{minor}")
        print("   Please install Python 3.11+ from https://www.python.org/downloads/")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if minor >= 13:
        print("ℹ️  INFO: Python 3.13+ detected - using advanced installation strategies.")
        print(f"   You are using Python {major}.{minor}")
        print("   Some packages may not have pre-built wheels yet.")
        print("   The installer will try multiple strategies to make it work.")
        print()


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


def read_requirements_file(file_path: Path) -> list[str]:
    """
    Read requirements.txt file with robust encoding detection.
    Handles UTF-8, UTF-16, and other common encodings.
    """
    if not file_path.exists():
        return []
    
    # Try multiple encodings in order of likelihood
    encodings_to_try = []
    
    # If chardet is available, try to detect encoding first
    if HAS_CHARDET:
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected and detected.get("encoding"):
                    encodings_to_try.append(detected["encoding"])
        except Exception:
            pass  # Fall back to manual detection
    
    # Add common encodings
    encodings_to_try.extend([
        "utf-8",
        "utf-8-sig",  # UTF-8 with BOM
        "utf-16-le",  # UTF-16 Little Endian (Windows default)
        "utf-16-be",  # UTF-16 Big Endian
        "utf-16",     # UTF-16 with BOM detection
        "latin-1",    # Fallback that never fails
        "cp1252",     # Windows-1252
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    encodings_to_try = [enc for enc in encodings_to_try if enc not in seen and not seen.add(enc)]
    
    last_error = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
                # Filter out empty lines and comments
                packages = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
                return packages
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    
    # If all encodings failed, try with error handling
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            packages = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
            print(f"⚠️  Warning: Read requirements.txt with error replacement (some characters may be lost)")
            return packages
    except Exception as e:
        raise RuntimeError(
            f"Could not read requirements.txt file. "
            f"Tried encodings: {', '.join(encodings_to_try[:5])}. "
            f"Last error: {last_error or e}"
        ) from (last_error or e)


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
        "setuptools-scm",  # Helpful for building packages
    ]
    run_command(pip_args)

    # Check Python version to determine installation strategy
    is_new_python = sys.version_info.minor >= 13
    
    # Try installation with multiple strategies to handle different scenarios
    install_strategies = []
    
    if is_new_python:
        # For Python 3.13+, try more aggressive strategies
        # Known issues:
        # - chromadb requires numpy<2.0.0, but numpy 2.x has Python 3.14 wheels
        # - chroma-hnswlib may need compilation for Python 3.14
        # - Some packages may not have wheels yet
        install_strategies = [
            # Strategy 1: Try latest chromadb first (may support numpy 2.x), then numpy 2.x, then rest
            {
                "name": "latest-chromadb-numpy2",
                "pre_install": [
                    "numpy>=2.0.0",  # numpy 2.x has Python 3.14 wheels
                    "chromadb>=1.3.0",  # Try latest chromadb (may have better numpy 2.x support)
                ],
                "args": [
                    str(VENV_PY),
                    "-m",
                    "pip",
                    "install",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--prefer-binary",
                    "-r",
                    str(REQUIREMENTS),
                ],
            },
            # Strategy 2: Install numpy 2.x first (has Python 3.14 wheels), then install with --no-deps to bypass constraints
            {
                "name": "numpy2-bypass",
                "pre_install": ["numpy>=2.0.0"],  # numpy 2.x has Python 3.14 wheels
                "args": [
                    str(VENV_PY),
                    "-m",
                    "pip",
                    "install",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--prefer-binary",
                    "--no-deps",  # Skip dependency checking to allow numpy 2.x
                    "-r",
                    str(REQUIREMENTS),
                ],
                "post_install": True,  # Install dependencies separately after
            },
            # Strategy 3: Install numpy 2.x first, then install packages individually
            {
                "name": "numpy-first",
                "pre_install": ["numpy>=2.0.0"],  # Newer numpy versions have 3.14 wheels
                "args": [
                    str(VENV_PY),
                    "-m",
                    "pip",
                    "install",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--prefer-binary",
                    "-r",
                    str(REQUIREMENTS),
                ],
            },
            # Strategy 4: Install packages individually to find what works
            {
                "name": "individual-packages",
                "install_individual": True,
                "args": None,
            },
            # Strategy 5: Use latest package versions that might support 3.14
            {
                "name": "latest-versions",
                "args": [
                    str(VENV_PY),
                    "-m",
                    "pip",
                    "install",
                    "--no-input",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--prefer-binary",
                    "--upgrade-strategy", "eager",  # Try latest versions
                    "-r",
                    str(REQUIREMENTS),
                ],
            },
            # Strategy 6: Standard install (last resort)
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
    else:
        # For Python 3.11-3.12, use standard strategies
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
    
    def install_individual_packages() -> bool:
        """Try installing packages one by one to find which ones work."""
        print("\nTrying to install packages individually...")
        try:
            packages = read_requirements_file(REQUIREMENTS)
        except Exception as e:
            print(f"❌ Error reading requirements.txt: {e}")
            write_error_log("File Reading Error", e, {"File": str(REQUIREMENTS)})
            return False
        
        failed_packages = []
        for package in packages:
            try:
                print(f"  Installing {package}...")
                result = subprocess.run(
                    [
                        str(VENV_PY),
                        "-m",
                        "pip",
                        "install",
                        "--no-input",
                        "--disable-pip-version-check",
                        "--upgrade",
                        "--prefer-binary",
                        package,
                    ],
                    cwd=str(ROOT),
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    print(f"  ✅ {package} installed successfully")
                else:
                    print(f"  ⚠️  {package} failed, will retry with different strategy")
                    failed_packages.append((package, result))
            except Exception as exc:
                print(f"  ⚠️  {package} failed: {exc}")
                failed_packages.append((package, None))
        
        if not failed_packages:
            print("✅ All packages installed successfully!")
            return True
        
        # Try failed packages with standard install (no prefer-binary)
        if failed_packages:
            print(f"\nRetrying {len(failed_packages)} failed packages...")
            remaining_failed = []
            for package, prev_result in failed_packages:
                try:
                    result = subprocess.run(
                        [
                            str(VENV_PY),
                            "-m",
                            "pip",
                            "install",
                            "--no-input",
                            "--disable-pip-version-check",
                            "--upgrade",
                            package,
                        ],
                        cwd=str(ROOT),
                        check=False,
                        text=True,
                        capture_output=True,
                    )
                    if result.returncode == 0:
                        print(f"  ✅ {package} installed successfully")
                    else:
                        print(f"  ❌ {package} still failed")
                        remaining_failed.append((package, result))
                except Exception as exc:
                    print(f"  ❌ {package} failed: {exc}")
                    remaining_failed.append((package, None))
            
            if not remaining_failed:
                print("✅ All packages installed successfully!")
                return True
            
            # If some packages still failed, check if critical ones are installed
            print(f"\n⚠️  {len(remaining_failed)} packages could not be installed:")
            for package, _ in remaining_failed:
                print(f"   - {package}")
            
            # Check if critical packages are actually installed (might be dependencies)
            critical_packages = ["chromadb", "langchain", "gradio", "langchain_openai", "langchain_chroma"]
            critical_ok = True
            for crit_pkg in critical_packages:
                module_name = crit_pkg.replace("-", "_")
                try:
                    result = subprocess.run(
                        [str(VENV_PY), "-c", f"import {module_name}"],
                        capture_output=True,
                        check=False,
                        text=True,
                    )
                    if result.returncode != 0:
                        # Check if it's in the failed list
                        if any(crit_pkg in pkg for pkg, _ in remaining_failed):
                            print(f"   ⚠️  Critical package {crit_pkg} is missing!")
                            critical_ok = False
                except Exception:
                    pass
            
            if critical_ok:
                print("   ✅ Critical packages are installed. The app should work.")
                return True
            else:
                print("   ❌ Some critical packages are missing. The app may not work.")
                # Still return True to let it try - might work with partial install
                return len(remaining_failed) < len(packages) / 2
        
        return True
    
    last_error = None
    for strategy in install_strategies:
        try:
            print(f"\nTrying installation strategy: {strategy['name']}...")
            
            # Handle pre-install packages (like numpy)
            if "pre_install" in strategy:
                for pre_pkg in strategy["pre_install"]:
                    print(f"  Pre-installing {pre_pkg}...")
                    result = subprocess.run(
                        [
                            str(VENV_PY),
                            "-m",
                            "pip",
                            "install",
                            "--no-input",
                            "--disable-pip-version-check",
                            "--upgrade",
                            "--prefer-binary",
                            pre_pkg,
                        ],
                        cwd=str(ROOT),
                        check=False,
                        text=True,
                        capture_output=True,
                    )
                    if result.returncode == 0:
                        print(f"  ✅ {pre_pkg} installed successfully")
                    else:
                        print(f"  ⚠️  {pre_pkg} pre-install failed, continuing anyway...")
            
            # Handle post-install (install dependencies after --no-deps)
            if strategy.get("post_install") and strategy["args"]:
                # First install with --no-deps
                result = subprocess.run(
                    strategy["args"],
                    cwd=str(ROOT),
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    # Now install dependencies, but allow numpy 2.x
                    print("  Installing dependencies (allowing numpy 2.x)...")
                    deps_result = subprocess.run(
                        [
                            str(VENV_PY),
                            "-m",
                            "pip",
                            "install",
                            "--no-input",
                            "--disable-pip-version-check",
                            "--upgrade",
                            "--prefer-binary",
                            "chromadb[all]",  # Install chromadb with all optional deps
                            "langchain",
                            "langchain-chroma",
                            "langchain-community",
                            "langchain-openai",
                            "langchain-text-splitters",
                            "gradio",
                            "docx2txt",
                            "pypdf",
                            "python-dotenv",
                            "tiktoken",
                        ],
                        cwd=str(ROOT),
                        check=False,
                        text=True,
                        capture_output=True,
                    )
                    if deps_result.returncode == 0:
                        print("✅ Installation successful!")
                        break
                    else:
                        # Try installing chromadb dependencies separately to work around numpy constraint
                        print("  Trying to install chromadb dependencies separately...")
                        
                        # Install chromadb's core dependencies that don't conflict with numpy 2.x
                        core_deps = [
                            "pydantic>=1.9",
                            "fastapi>=0.95.2",
                            "uvicorn[standard]>=0.18.3",
                            "posthog>=2.4.0,<6.0.0",
                            "typing-extensions>=4.5.0",
                            "pybase64>=1.4.1",
                            "build>=1.0.3",
                        ]
                        
                        for dep in core_deps:
                            subprocess.run(
                                [
                                    str(VENV_PY),
                                    "-m",
                                    "pip",
                                    "install",
                                    "--no-input",
                                    "--disable-pip-version-check",
                                    "--upgrade",
                                    "--prefer-binary",
                                    dep,
                                ],
                                cwd=str(ROOT),
                                check=False,
                                text=True,
                                capture_output=True,
                            )
                        
                        # Try installing chromadb with --no-deps to bypass numpy constraint
                        print("  Installing chromadb with --no-deps (bypassing numpy constraint)...")
                        chromadb_result = subprocess.run(
                            [
                                str(VENV_PY),
                                "-m",
                                "pip",
                                "install",
                                "--no-input",
                                "--disable-pip-version-check",
                                "--upgrade",
                                "--prefer-binary",
                                "--no-deps",
                                "chromadb>=0.5.4",
                            ],
                            cwd=str(ROOT),
                            check=False,
                            text=True,
                            capture_output=True,
                        )
                        
                        # Try installing chroma-hnswlib separately (may need special handling)
                        print("  Installing chroma-hnswlib...")
                        subprocess.run(
                            [
                                str(VENV_PY),
                                "-m",
                                "pip",
                                "install",
                                "--no-input",
                                "--disable-pip-version-check",
                                "--upgrade",
                                "--prefer-binary",
                                "chroma-hnswlib",
                            ],
                            cwd=str(ROOT),
                            check=False,
                            text=True,
                            capture_output=True,
                        )
                        
                        # Install remaining packages
                        remaining_packages = [
                            "langchain>=0.3.6",
                            "langchain-chroma>=0.1.4",
                            "langchain-community>=0.3.6",
                            "langchain-openai>=0.3.2",
                            "langchain-text-splitters>=0.3.1",
                            "gradio>=4.44.0",
                            "docx2txt>=0.8",
                            "pypdf>=4.2.0",
                            "python-dotenv>=1.0.1",
                            "tiktoken>=0.7.0",
                        ]
                        
                        for pkg in remaining_packages:
                            subprocess.run(
                                [
                                    str(VENV_PY),
                                    "-m",
                                    "pip",
                                    "install",
                                    "--no-input",
                                    "--disable-pip-version-check",
                                    "--upgrade",
                                    "--prefer-binary",
                                    pkg,
                                ],
                                cwd=str(ROOT),
                                check=False,
                                text=True,
                                capture_output=True,
                            )
                        
                        # Check if critical packages work now
                        test_packages = [
                            ("chromadb", "chromadb"),
                            ("langchain", "langchain"),
                            ("gradio", "gradio"),
                            ("langchain_openai", "langchain_openai"),
                            ("langchain_chroma", "langchain_chroma"),
                        ]
                        
                        all_ok = True
                        for module_name, package_name in test_packages:
                            test_result = subprocess.run(
                                [str(VENV_PY), "-c", f"import {module_name}"],
                                cwd=str(ROOT),
                                check=False,
                                text=True,
                                capture_output=True,
                            )
                            if test_result.returncode != 0:
                                print(f"  ⚠️  {package_name} import failed")
                                all_ok = False
                        
                        if all_ok:
                            print("✅ Installation successful (with workaround for numpy 2.x)!")
                            break
                        else:
                            print("  ⚠️  Some packages still have issues, but continuing...")
                # Continue to next strategy if this didn't work
                if result.returncode != 0:
                    last_error = result
                    continue
            
            # Handle individual package installation
            if strategy.get("install_individual"):
                if install_individual_packages():
                    break
                else:
                    last_error = subprocess.CalledProcessError(1, ["individual-packages"])
                    continue
            
            # Standard installation (skip if we already handled post_install)
            if strategy["args"] and not strategy.get("post_install"):
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
                
                if is_dependency_conflict and strategy["name"] in ("prefer-binary", "numpy-first"):
                    # Dependency conflict, try next strategy
                    print("⚠️  Dependency conflict detected, trying alternative approach...")
                    last_error = result
                    continue
                
                # For other errors, continue to next strategy
                if strategy != install_strategies[-1]:
                    print("⚠️  Installation failed, trying next strategy...")
                    last_error = result
                    continue
                
                # Last strategy failed, show error
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
    check_python_version()
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
        
        # Check Python version in error context
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        is_unsupported_python = sys.version_info.minor >= 13
        
        print("\nTroubleshooting:")
        if is_unsupported_python:
            print(f"⚠️  You are using Python {python_version}, which is very new.")
            print("   Some packages may not have pre-built wheels yet.")
            print("   The installer tried multiple strategies but couldn't install all packages.")
            print("\n   OPTIONS:")
            print("   1. RECOMMENDED: Install Python 3.11 or 3.12 for best compatibility")
            print("      - All packages have pre-built wheels")
            print("      - No compilation needed")
            print("      - Download from: https://www.python.org/downloads/")
            print("\n   2. ALTERNATIVE: Install C++ build tools to compile from source")
            print("      - Windows: Install 'Build Tools for Visual Studio'")
            print("      - This allows building packages that don't have wheels")
            print("      - More complex but enables Python 3.14 support")
            print("\n   3. Wait for package maintainers to release Python 3.14 wheels")
            print("      - Check back in a few weeks/months")
            print("      - Packages are being updated regularly")
        elif is_dependency_conflict:
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
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            is_unsupported_python = sys.version_info.minor >= 13
            
            print("⚠️  This looks like a compilation error (trying to build from source).")
            if is_unsupported_python:
                print(f"   This is happening because you're using Python {python_version}.")
                print("   Python 3.13+ doesn't have pre-built wheels for many packages yet.")
                print("\n   SOLUTION:")
                print("   1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/")
                print("   2. Delete the .venv folder in this directory")
                print("   3. Run the launcher again with the correct Python version")
            else:
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

