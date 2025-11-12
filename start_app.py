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
    
    # Add Python version info
    if sys.version_info.minor >= 13:
        info.append(f"\n✅ Python {sys.version_info.major}.{sys.version_info.minor} is fully supported!")
        info.append("   Using optimized package versions for this Python version.")
    
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
    """Check if Python version is supported and warn if not."""
    version = sys.version_info
    major, minor = version.major, version.minor
    
    if major != 3:
        print("\n" + "=" * 68)
        print("❌ ERROR: Python 3 is required".center(68))
        print("=" * 68)
        print(f"\n   You are using Python {major}.{minor}.{version.micro}")
        print("\n   SOLUTION:")
        print("   1. Download Python 3.13.3, 3.12, or 3.14 from:")
        print("      https://www.python.org/downloads/")
        print("\n   2. During installation, CHECK THIS BOX:")
        print("      ☑ 'Add Python to PATH' (very important!)")
        print("\n   3. After installing, close this window and try again")
        print("\n   Need help? See GETTING_STARTED.md for detailed instructions.")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if minor < 11:
        print("\n" + "=" * 68)
        print("❌ ERROR: Python 3.11 or newer is required".center(68))
        print("=" * 68)
        print(f"\n   You are using Python {major}.{minor}.{version.micro}")
        print("   This version is too old for this application.")
        print("\n   SOLUTION:")
        print("   1. Download Python 3.13.3, 3.12, or 3.14 from:")
        print("      https://www.python.org/downloads/")
        print("\n   2. During installation, CHECK THIS BOX:")
        print("      ☑ 'Add Python to PATH' (very important!)")
        print("\n   3. After installing, close this window and try again")
        input("\nPress Enter to close...")
        sys.exit(1)
    
    if minor >= 13:
        # Python 3.13+ is now well-supported (November 2025)
        print(f"\n✅ Python {major}.{minor}.{version.micro} detected")
        print("   This version is fully supported!")
        if minor >= 14:
            print("   Python 3.14 detected - using latest optimized packages.\n")
        else:
            print("   Python 3.13 detected - using optimized packages.\n")


def check_network_connectivity() -> bool:
    """Check if we can reach the internet (needed for pip installs)."""
    import socket
    import urllib.request
    import urllib.error
    
    print("Checking internet connection...", end=" ", flush=True)
    
    # Try multiple methods
    test_hosts = [
        ("8.8.8.8", 53),  # Google DNS
        ("1.1.1.1", 53),  # Cloudflare DNS
        ("pypi.org", 443),  # PyPI
    ]
    
    for host, port in test_hosts:
        try:
            socket.create_connection((host, port), timeout=3)
            print("✅ Connected")
            return True
        except (socket.error, OSError, socket.timeout):
            continue
    
    # Try HTTP request as fallback
    try:
        urllib.request.urlopen("https://pypi.org", timeout=5)
        print("✅ Connected")
        return True
    except (urllib.error.URLError, Exception):
        pass
    
    print("❌ No connection")
    return False


def check_disk_space() -> bool:
    """Check if we have enough disk space (need at least 500MB)."""
    import shutil
    
    print("Checking disk space...", end=" ", flush=True)
    try:
        free_bytes = shutil.disk_usage(ROOT).free
        free_mb = free_bytes / (1024 * 1024)
        min_required_mb = 500
        
        if free_mb < min_required_mb:
            print(f"❌ Only {free_mb:.0f} MB free (need {min_required_mb} MB)")
            return False
        print(f"✅ {free_mb:.0f} MB available")
        return True
    except Exception as e:
        print(f"⚠️  Could not check ({e})")
        return True  # Assume OK if we can't check


def check_write_permissions() -> bool:
    """Check if we can write to the project directory."""
    print("Checking write permissions...", end=" ", flush=True)
    try:
        test_file = ROOT / ".write_test"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        print("✅ OK")
        return True
    except PermissionError:
        print("❌ No permission")
        return False
    except Exception as e:
        print(f"⚠️  Could not check ({e})")
        return True  # Assume OK if we can't check


def check_windows_path_length() -> bool:
    """Check if we're on Windows and path might be too long."""
    if os.name != "nt":
        return True  # Not Windows, no issue
    
    print("Checking path length...", end=" ", flush=True)
    try:
        # Windows has a 260 character path limit by default
        full_path = str(ROOT.resolve())
        if len(full_path) > 200:  # Warn if getting close
            print(f"⚠️  Path is {len(full_path)} chars (may cause issues)")
            print(f"   Consider moving the project to a shorter path like C:\\RAG")
            return True  # Still allow, just warn
        print("✅ OK")
        return True
    except Exception:
        print("⚠️  Could not check")
        return True


def run_preflight_checks() -> bool:
    """Run all pre-flight checks before starting installation."""
    print("\n" + "=" * 68)
    print(" Running pre-flight checks...".center(68))
    print("=" * 68 + "\n")
    
    checks = [
        ("Network connectivity", check_network_connectivity),
        ("Disk space", check_disk_space),
        ("Write permissions", check_write_permissions),
        ("Path length", check_windows_path_length),
    ]
    
    failed_checks = []
    for name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(name)
        except Exception as e:
            print(f"⚠️  {name} check failed: {e}")
    
    if failed_checks:
        print("\n" + "=" * 68)
        print("⚠️  Some checks failed".center(68))
        print("=" * 68)
        print("\n   The following issues were detected:")
        for check in failed_checks:
            print(f"   - {check}")
        
        if "Network connectivity" in failed_checks:
            print("\n   SOLUTION for Network:")
            print("   - Check your internet connection")
            print("   - Check if a firewall is blocking connections")
            print("   - Try again when you have internet access")
        
        if "Write permissions" in failed_checks:
            print("\n   SOLUTION for Permissions:")
            print("   - Move the project folder to a location where you have")
            print("     write permissions (like Documents or Desktop)")
            print("   - Or run this as Administrator (right-click → Run as administrator)")
        
        if "Disk space" in failed_checks:
            print("\n   SOLUTION for Disk Space:")
            print("   - Free up at least 500 MB of disk space")
            print("   - Delete unnecessary files or move the project to another drive")
        
        response = input("\n   Continue anyway? (y/n): ").strip().lower()
        if response not in ('y', 'yes'):
            print("\n   Installation cancelled. Please fix the issues above.")
            input("\nPress Enter to close...")
            return False
    
    print("\n✅ All checks passed!\n")
    return True


def print_header() -> None:
    print("=" * 68)
    print(" Document RAG Assistant launcher".center(68))
    print("=" * 68)
    print("\nThis will set up everything automatically. Please wait...")
    print("(This may take 2-5 minutes the first time)\n")


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    quiet: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"\n$ {' '.join(args)}")
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd or ROOT),
            check=False,  # Don't raise on error, we'll handle it
            text=True,
            capture_output=capture_output,
            timeout=timeout,  # Add timeout to prevent hanging
        )
    except subprocess.TimeoutExpired:
        exc = subprocess.CalledProcessError(1, args)
        exc.stdout = ""
        exc.stderr = f"Command timed out after {timeout} seconds"
        raise exc
    
    if result.returncode != 0:
        # Re-raise with captured output for better error messages
        exc = subprocess.CalledProcessError(result.returncode, args)
        exc.stdout = result.stdout if capture_output else ""
        exc.stderr = result.stderr if capture_output else ""
        raise exc
    return result


def run_command_with_capture(
    args: list[str],
    *,
    cwd: Path | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run a command with real-time output display AND capture output for error logging.
    Uses a simple, reliable approach: capture output and print it line-by-line.
    Combines stdout and stderr since pip outputs errors to both.
    """
    if not quiet:
        print(f"\n$ {' '.join(args)}")
    
    # Use Popen to read line-by-line and print in real-time
    # Combine stderr into stdout since pip outputs errors to both
    process = subprocess.Popen(
        args,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture everything
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )
    
    # Capture all output lines
    output_lines = []
    
    # Read and print output line by line
    try:
        # Read all output line by line (this blocks until process completes or pipe closes)
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip('\n\r')
                output_lines.append(line)
                print(line)  # Print in real-time
                sys.stdout.flush()  # Ensure immediate display
        
        # Wait for process to complete (should already be done, but ensure)
        return_code = process.wait()
        
        # Close the pipe
        process.stdout.close()
        
    except BrokenPipeError:
        # Process closed stdout, wait for it to finish
        return_code = process.wait()
    except Exception as e:
        # If reading fails, wait for process and capture what we can
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Process hung, kill it
            try:
                process.kill()
                process.wait(timeout=5)
            except:
                pass
            return_code = 1
            output_lines.append(f"Process timed out or error reading output: {e}")
        except:
            return_code = 1
            output_lines.append(f"Error reading process output: {e}")
    
    # Combine all output (stdout + stderr since we redirected)
    full_output = '\n'.join(output_lines)
    
    # Create result object
    result = subprocess.CompletedProcess(
        args,
        return_code,
        stdout=full_output,
        stderr="",  # Already included in stdout
    )
    
    if return_code != 0:
        # Create exception with full output in both stdout and stderr
        # This ensures error detection works regardless of where we check
        exc = subprocess.CalledProcessError(return_code, args)
        exc.stdout = full_output
        exc.stderr = full_output  # Put in stderr too for compatibility
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
        print("✅ Virtual environment already exists")
        return
    print("\n📦 Step 1/3: Creating virtual environment...")
    print("   (This creates an isolated Python environment for this project)")
    try:
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT)
        print("   ✅ Virtual environment created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create virtual environment: {e}")
        raise


def ensure_dependencies() -> None:
    fingerprint = requirements_fingerprint()
    missing = modules_missing(VENV_PY)
    if not missing and fingerprint is not None and STAMP_FILE.exists():
        saved = STAMP_FILE.read_text(encoding="utf-8").strip()
        if saved == fingerprint:
            print("\nDependencies already satisfied.")
            return

    print("\n📦 Step 2/3: Installing dependencies...")
    print("   (This downloads and installs required packages)")
    print("   This may take 3-10 minutes depending on your internet speed.")
    print("   Large packages like chromadb (~100MB) can take 1-2 minutes to download.")
    print("   You'll see progress below - please be patient!\n")
    
    print("   Upgrading pip and build tools...")
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
    try:
        run_command(pip_args)
        print("   ✅ Build tools upgraded")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not upgrade build tools: {e}")
        print("   Continuing anyway...")

    # Check Python version to determine installation strategy
    # Python 3.13+ is now well-supported (November 2025), so we use standard strategies
    is_new_python = sys.version_info.minor >= 14  # Only use special strategies for 3.14+
    
    # Try installation with multiple strategies to handle different scenarios
    install_strategies = []
    
    if is_new_python:
        # For Python 3.14+, use optimized strategies
        # Most packages now have wheels for 3.14, but we keep fallback strategies
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
        # For Python 3.11-3.14, use standard strategies (all well-supported now)
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
            print(f"\n{'=' * 68}")
            print(f"Trying installation strategy: {strategy['name']}".center(68))
            print("=" * 68)
            
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
                print("   Installing packages with --no-deps (this may take a few minutes)...")
                print("   Please be patient - you'll see progress below:\n")
                # First install with --no-deps
                try:
                    result = run_command_with_capture(
                        strategy["args"],
                        cwd=ROOT,
                        quiet=True,
                    )
                except subprocess.CalledProcessError as exc:
                    result = subprocess.CompletedProcess(
                        strategy["args"],
                        exc.returncode,
                        exc.stdout or "",
                        exc.stderr or "",
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
                        print("\n" + "=" * 68)
                        print("✅ Installation successful!".center(68))
                        print("=" * 68 + "\n")
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
                            print("\n" + "=" * 68)
                            print("✅ Installation successful (with workaround)!".center(68))
                            print("=" * 68 + "\n")
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
                print("   Installing packages (this may take a few minutes)...")
                print("   Large packages like chromadb can take 1-2 minutes to download.")
                print("   Please be patient - you'll see progress below:\n")
                
                # Run with real-time output AND capture for error logging
                try:
                    result = run_command_with_capture(
                        strategy["args"],
                        cwd=ROOT,
                        quiet=True,  # Don't print command again
                    )
                    if result.returncode == 0:
                        print("\n" + "=" * 68)
                        print("✅ Installation successful!".center(68))
                        print("=" * 68 + "\n")
                        break
                except subprocess.CalledProcessError as exc:
                    # Installation failed - we have captured output
                    print(f"\n⚠️  Installation strategy '{strategy['name']}' failed (exit code: {exc.returncode})")
                    
                    # Get error output from either stdout or stderr (we capture both)
                    error_output = ""
                    if hasattr(exc, 'stdout') and exc.stdout:
                        error_output = exc.stdout
                    elif hasattr(exc, 'stderr') and exc.stderr:
                        error_output = exc.stderr
                    
                    # Show last few lines of error for user
                    if error_output and error_output.strip():
                        error_lines = error_output.strip().split('\n')
                        if len(error_lines) > 0:
                            print("\n   Last error messages:")
                            # Show last 10 lines (more than before)
                            for line in error_lines[-10:]:
                                if line.strip():
                                    print(f"   {line}")
                    
                    # For other errors, continue to next strategy
                    if strategy != install_strategies[-1]:
                        print("   Trying next strategy...\n")
                        last_error = exc
                        continue
                    
                    # Last strategy failed
                    print("\n❌ All installation strategies failed.")
                    print("   Check the error messages above for details.")
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
    print("📦 Step 3/3: Launching the assistant...".center(68))
    print("=" * 68)
    print("\n✅ Setup complete! The app will open in your browser shortly.")
    print("\n   IMPORTANT:")
    print("   - Keep this window open while using the app")
    print("   - Press Ctrl+C to stop the app when you're done")
    print("   - The first time may take a moment to load\n")
    
    # Add antivirus warning for Windows
    if os.name == "nt":
        print("   💡 TIP: If Windows Defender or antivirus shows a warning,")
        print("      click 'Allow' or 'More info' → 'Run anyway'")
        print("      This is normal for Python applications.\n")
    
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.call([str(python_for_app), str(ROOT / "app.py")], cwd=str(ROOT))


def main() -> None:
    check_python_version()
    print_header()
    
    # Run pre-flight checks
    if not run_preflight_checks():
        sys.exit(1)
    
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
        
        # Show error output if available (check both stdout and stderr)
        error_output_text = ""
        if hasattr(exc, 'stdout') and exc.stdout and exc.stdout.strip():
            error_output_text = exc.stdout
        elif hasattr(exc, 'stderr') and exc.stderr and exc.stderr.strip():
            error_output_text = exc.stderr
        
        if error_output_text:
            print("\n" + "=" * 68)
            print(" Error Output:".center(68))
            print("=" * 68)
            # Show last 30 lines to avoid overwhelming the user but show enough context
            error_lines = error_output_text.strip().split('\n')
            if len(error_lines) > 30:
                print("   (Showing last 30 lines of error output)")
                print("   (Full error saved to error_log.txt)\n")
                for line in error_lines[-30:]:
                    if line.strip():
                        print(f"   {line}")
            else:
                for line in error_lines:
                    if line.strip():
                        print(f"   {line}")
            print("=" * 68)
        
        # Log the error - combine stdout and stderr for analysis
        error_output = ""
        if hasattr(exc, 'stdout') and exc.stdout:
            error_output = exc.stdout.lower()
        if hasattr(exc, 'stderr') and exc.stderr:
            if error_output:
                error_output += " " + exc.stderr.lower()
            else:
                error_output = exc.stderr.lower()
        
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
        
        print("\n" + "=" * 68)
        print(" Troubleshooting Guide".center(68))
        print("=" * 68)
        
        if is_unsupported_python:
            print(f"\n⚠️  You are using Python {python_version}")
            if python_version >= "3.14":
                print("   Python 3.14 is very new. Most packages should work, but if you")
                print("   encounter issues, try Python 3.13.3 or 3.12 for maximum compatibility.")
            else:
                print("   The installer tried multiple strategies but couldn't install all packages.")
            print("\n   SOLUTIONS:")
            print("   1. Try deleting the .venv folder and running again")
            print("      (Sometimes a fresh install fixes issues)")
            print("\n   2. If still failing, try Python 3.13.3 or 3.12:")
            print("      - Download from: https://www.python.org/downloads/")
            print("      - During installation, CHECK THIS BOX:")
            print("        ☑ 'Add Python to PATH'")
            print("      - Delete .venv folder and run start_app.bat again")
            print("\n   3. Check your internet connection")
            print("      - Make sure you can access pypi.org")
            print("      - Try disabling VPN if you're using one")
        elif is_dependency_conflict:
            print("\n⚠️  Dependency conflict detected")
            print("   (Some packages have incompatible requirements)")
            print("\n   SOLUTIONS (try in order):")
            print("   1. Delete the .venv folder in this directory and try again")
            print("      (The installer will recreate it)")
            print("\n   2. Make sure you're using Python 3.11 or 3.12")
            print("      - Check: python --version")
            print("      - If not 3.11 or 3.12, install from python.org")
            print("\n   3. Check your internet connection")
            print("      - Make sure you can access pypi.org")
            print("      - Try disabling VPN if you're using one")
            print("\n   4. If still failing:")
            print("      - Reinstall Python from python.org")
            print("      - Make sure to check 'Add Python to PATH' during install")
            print("      - Delete .venv folder and try again")
        elif is_compilation_error:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            is_unsupported_python = sys.version_info.minor >= 13
            
            print("\n⚠️  Compilation error detected")
            print("   (The installer tried to build packages from source code)")
            if is_unsupported_python:
                print(f"\n   CAUSE: You're using Python {python_version}")
                if python_version >= "3.14":
                    print("   Python 3.14 is very new. Most packages should work, but if you")
                    print("   encounter issues, try Python 3.13.3 or 3.12 for maximum compatibility.")
                else:
                    print("   Most packages now support Python 3.13, but some may still need compilation.")
            else:
                print("\n   This usually happens when:")
                print("   - Pre-built installers aren't available for your system")
                print("   - Python version is too new or too old")
            print("\n   SOLUTIONS:")
            print("   1. Try deleting the .venv folder and running again")
            print("      (Sometimes a fresh install fixes issues)")
            print("\n   2. Make sure you have Python 3.11, 3.12, 3.13, or 3.14")
            print("      - Check: python --version")
            print("      - If not a supported version, install from python.org")
            print("\n   3. If still failing, try Python 3.13.3 or 3.12:")
            print("      - Download from: https://www.python.org/downloads/")
            print("      - During installation, CHECK THIS BOX:")
            print("        ☑ 'Add Python to PATH'")
            print("      - Delete .venv folder and run start_app.bat again")
            print("\n   4. Check your internet connection")
            print("      - Make sure you can access pypi.org")
            print("      - Try disabling VPN if you're using one")
        else:
            print("\n   GENERAL SOLUTIONS:")
            print("   1. Make sure Python 3.11, 3.12, 3.13, or 3.14 is installed")
            print("      - Check: python --version")
            print("      - If not installed, get it from python.org")
            print("\n   2. Check your internet connection")
            print("      - Make sure you can access pypi.org")
            print("      - Try disabling VPN if you're using one")
            print("\n   3. Delete the .venv folder and try again")
            print("      (The installer will recreate it)")
            print("\n   4. Check antivirus/firewall")
            print("      - Windows Defender or antivirus may be blocking")
            print("      - Try adding this folder to exclusions")
            print("\n   5. Try running again")
            print("      - Sometimes it's a temporary network issue")
        
        print("\n" + "=" * 68)
        print(f"📝 Error details saved to: {ERROR_LOG}".center(68))
        print("=" * 68)
        print("\n   If you need help, share this file with someone who can help.")
        print("   The file contains technical details about what went wrong.")
        
        if os.name == "nt":
            print("\n   💡 TIP: If Windows Defender blocked something,")
            print("      click 'Allow' or 'More info' → 'Run anyway'")
            print("      This is normal for Python applications.")
        
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

