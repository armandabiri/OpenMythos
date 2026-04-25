"""Run absolufy-imports, isort, and ruff on Python files under a path."""

import argparse
import concurrent.futures
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE = ".*,.mypy_cache,.venv"
ISORT_CFG_NAME = ".isort.cfg"
ISORT_CFG_CONTENT = """[settings]
profile = black
import_heading_stdlib = Standard library imports
import_heading_thirdparty = Third party imports
import_heading_firstparty = Internal imports
"""
EXCLUDED_DIRS = ("build", "intelag_packages")
DEPS_FORMAT_LINT = ["isort", "ruff", "absolufy-imports"]


def get_python_files(root_path: Path) -> list[Path]:
    """Return list of Python file paths under root_path, excluding hidden and excluded dirs."""
    found_files: list[Path] = []
    for path in root_path.rglob("*.py"):
        parts = path.parts
        if (
            not any(p.startswith(".") for p in parts)
            and EXCLUDED_DIRS[0] not in parts
            and EXCLUDED_DIRS[1] not in parts
        ):
            found_files.append(path)
    return found_files


def run_absolufy(file_path: Path) -> subprocess.CompletedProcess[str]:
    """Run absolufy-imports on a single file path."""
    return subprocess.run(
        ["absolufy-imports", str(file_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    """Install deps and run absolufy, isort, ruff on the given path."""
    parser = argparse.ArgumentParser(description="Intelag Format and Lint")
    parser.add_argument(
        "--python-exe",
        required=True,
        help="Path to python executable",
    )
    parser.add_argument("--path", default=".", help="Path to scan")
    parser.add_argument(
        "--exclude",
        default=DEFAULT_EXCLUDE,
        help="Exclude patterns",
    )

    args = parser.parse_args()
    python_exe = args.python_exe
    root_path = Path(args.path).resolve()

    logger.info("Formatting and linting: %s", root_path)

    subprocess.run(
        [python_exe, "-m", "pip", "install", *DEPS_FORMAT_LINT],
        check=False,
    )

    files = get_python_files(root_path)

    logger.info("Absolufy imports...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(run_absolufy, files))

    logger.info("Sorting imports...")
    isort_cfg = root_path / ISORT_CFG_NAME
    try:
        with isort_cfg.open("w", encoding="utf-8") as f:
            f.write(ISORT_CFG_CONTENT)
        subprocess.run([python_exe, "-m", "isort", str(root_path)], check=False)
    finally:
        if isort_cfg.exists():
            isort_cfg.unlink()

    logger.info("Running ruff...")
    subprocess.run(
        [
            python_exe,
            "-m",
            "ruff",
            "check",
            str(root_path),
            "--fix",
            "--exclude",
            args.exclude,
        ],
        check=False,
    )

    logger.info("Format and lint complete.")


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: format_and_lint.py --python-exe python --path .")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
