"""Run monkeytype, absolufy, autoflake, isort, ruff, vulture, docformatter on a path."""

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
ISSUES_DIR = ".intelag/issues"
VULTURE_REPORT = ".intelag/issues/vulture_report.txt"
DEPS_FULL_CHECK = [
    "monkeytype",
    "isort",
    "ruff",
    "absolufy-imports",
    "vulture",
    "docformatter",
    "autoflake",
]


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


def module_name_from_file(file_path: Path, root_path: Path) -> str:
    """Convert a Python file path to a module path for monkeytype apply."""
    module_parts = list(file_path.relative_to(root_path).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def run_monkeytype_apply(python_exe: str, root_path: Path, file_path: Path) -> subprocess.CompletedProcess[str]:
    """Run monkeytype apply for one file."""
    module_name = module_name_from_file(file_path, root_path)
    return subprocess.run(
        [python_exe, "-m", "monkeytype", "apply", module_name],
        check=False,
        capture_output=True,
        text=True,
    )


def run_absolufy(file_path: Path) -> subprocess.CompletedProcess[str]:
    """Run absolufy-imports for one file."""
    return subprocess.run(
        ["absolufy-imports", str(file_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def run_monkeytype_apply_for_all(
    python_exe: str,
    root_path: Path,
    py_files: list[Path],
) -> None:
    """Run monkeytype apply for all discovered files."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_monkeytype_apply, python_exe, root_path, file_path) for file_path in py_files]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    """Install deps and run full check pipeline on the given path."""
    parser = argparse.ArgumentParser(description="Intelag Full Project Check")
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

    logger.info("Starting full check on: %s", root_path)

    subprocess.run(
        [python_exe, "-m", "pip", "install", *DEPS_FULL_CHECK],
        check=False,
    )

    py_files = get_python_files(root_path)

    logger.info("Applying types to %d files...", len(py_files))
    run_monkeytype_apply_for_all(python_exe, root_path, py_files)

    logger.info("Converting to absolute imports...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(run_absolufy, py_files))

    logger.info("Cleaning unused imports/variables...")
    subprocess.run(
        [
            python_exe,
            "-m",
            "autoflake",
            "--in-place",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--recursive",
            str(root_path),
            "--exclude",
            args.exclude,
        ],
        check=False,
    )

    logger.info("Sorting imports...")
    isort_cfg = root_path / ISORT_CFG_NAME
    try:
        with isort_cfg.open("w", encoding="utf-8") as f:
            f.write(ISORT_CFG_CONTENT)
        subprocess.run([python_exe, "-m", "isort", str(root_path)], check=False)
    finally:
        if isort_cfg.exists():
            isort_cfg.unlink()

    logger.info("Running ruff linter...")
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

    logger.info("Finding dead code...")
    issues_dir_path = Path(ISSUES_DIR)
    issues_dir_path.mkdir(parents=True, exist_ok=True)
    vulture_report_path = Path(VULTURE_REPORT)
    with vulture_report_path.open("w", encoding="utf-8") as f:
        subprocess.run(
            [
                python_exe,
                "-m",
                "vulture",
                str(root_path),
                "--exclude",
                args.exclude,
            ],
            stdout=f,
            check=False,
        )

    logger.info("Formatting docstrings...")
    subprocess.run(
        [
            python_exe,
            "-m",
            "docformatter",
            "--in-place",
            "--recursive",
            str(root_path),
            "--exclude",
            args.exclude,
            "--wrap-summaries",
            "0",
            "--wrap-descriptions",
            "0",
            "--make-summary-multi-line",
        ],
        check=False,
    )

    logger.info("Full check complete.")


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: full_check.py --python-exe python --path .")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
