"""Apply MonkeyType-generated type stubs to Python files; optionally run a script to collect types."""

import argparse
import concurrent.futures
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_DIRS = (
    "build",
    "intelag_packages",
    "intelag_data_collection_manager",
    "intelag_vsix_packages",
    "intelag_vsix_creator",
    "common_media",
    "script",
)


def get_python_files(root_path: Path) -> list[Path]:
    """Return list of Python file paths under root_path, excluding hidden and excluded dirs."""
    found_files: list[Path] = []
    for path in root_path.rglob("*.py"):
        parts = path.parts
        if not any(p.startswith(".") for p in parts) and not any(d in parts for d in DEFAULT_EXCLUDE_DIRS):
            found_files.append(path)
    return found_files


def module_name_from_file(file_path: Path, root_path: Path) -> str:
    """Convert a Python file path to a module path for monkeytype apply."""
    module_parts = list(file_path.relative_to(root_path).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def run_monkeytype_apply(
    python_exe: str,
    root_path: Path,
    file_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Run monkeytype apply for one file."""
    return subprocess.run(
        [
            python_exe,
            "-m",
            "monkeytype",
            "apply",
            module_name_from_file(file_path, root_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def run_monkeytype_apply_for_all(
    python_exe: str,
    root_path: Path,
    files: list[Path],
) -> None:
    """Run monkeytype apply concurrently for all discovered files."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_monkeytype_apply, python_exe, root_path, file_path) for file_path in files]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    """Install monkeytype, optionally run collection script, then apply types to all Python files."""
    parser = argparse.ArgumentParser(description="Intelag MonkeyType Apply")
    parser.add_argument(
        "--python-exe",
        required=True,
        help="Path to python executable",
    )
    parser.add_argument("--path", default=".", help="Path to scan")
    parser.add_argument(
        "--script",
        default=None,
        help="Script to run for type collection (optional)",
    )
    args = parser.parse_args()

    python_exe = args.python_exe
    root_path = Path(args.path).resolve()

    subprocess.run(
        [python_exe, "-m", "pip", "install", "monkeytype"],
        check=False,
    )

    if args.script:
        logger.info("Running type collection script: %s", args.script)
        subprocess.run(
            [python_exe, "-m", "monkeytype", "run", args.script],
            check=False,
        )

    files = get_python_files(root_path)
    logger.info("Applying types to %d files...", len(files))

    run_monkeytype_apply_for_all(python_exe, root_path, files)

    logger.info("MonkeyType apply complete.")


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: monkeytype_apply.py --python-exe python --path .")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
