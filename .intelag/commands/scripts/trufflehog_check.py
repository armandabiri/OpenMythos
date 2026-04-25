"""Run TruffleHog deep git scan for secrets (optionally with entropy checks)."""

import argparse
import datetime
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "trufflehog"
DEFAULT_MAX_DEPTH = 1000000
FILE_URI_PREFIX = "file:///"
REPORT_SEP = "-" * 40


def _find_trufflehog_exe() -> Optional[str]:
    """Locate trufflehog executable or return None."""
    exe = shutil.which(TOOL_NAME)
    if exe:
        return exe
    scripts_dir = Path(sys.executable).parent
    suffix = "trufflehog.exe" if sys.platform == "win32" else "trufflehog"
    candidate = scripts_dir / suffix
    if candidate.exists():
        return str(candidate)
    return None


def _install_tool(tool_name: str) -> None:
    """Install a Python tool via pip; exit on failure."""
    logger.info("Installing %s...", tool_name)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", tool_name])
        logger.info("Successfully installed %s.", tool_name)
    except subprocess.CalledProcessError as e:
        logger.error("Installing %s failed: %s", tool_name, e)
        sys.exit(1)


def main() -> None:
    """Run trufflehog on path or repo; optionally write report to file."""
    parser = argparse.ArgumentParser(description="Run TruffleHog deep git scanning.")
    parser.add_argument(
        "--path",
        default=".",
        help="Path to scan (git repo URL or file path)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for the report",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        help="Enable entropy checks",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Max commit depth",
    )
    args = parser.parse_args()

    trufflehog_exe = _find_trufflehog_exe()
    if not trufflehog_exe:
        logger.info("%s not found. Attempting to install...", TOOL_NAME)
        _install_tool(TOOL_NAME)
        trufflehog_exe = _find_trufflehog_exe()
    if not trufflehog_exe:
        trufflehog_exe = TOOL_NAME

    repo_path = args.path
    if repo_path == ".":
        repo_path = FILE_URI_PREFIX + str(Path.cwd()).replace("\\", "/")
    elif not repo_path.startswith("file://") and not repo_path.startswith("http"):
        repo_path = FILE_URI_PREFIX + str(Path(repo_path).resolve()).replace("\\", "/")

    cmd: List[str] = [trufflehog_exe, "--regex"]
    if args.entropy:
        cmd.append("--entropy")
        cmd.append("True")
    cmd.extend(["--max_depth", str(args.max_depth)])
    cmd.append(repo_path)

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0 and not stdout:
            logger.error("Error: %s", stderr)
        logger.info("%s", stdout)

        if args.output:
            output_path = Path(args.output)
            out_dir = output_path.parent
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Date: {datetime.datetime.now()}\n")
                f.write(REPORT_SEP + "\n\n")
                if not stdout:
                    f.write("No secrets found (or error occurred).\n")
                    if stderr:
                        f.write("\nError Log:\n" + stderr)
                else:
                    f.write(stdout)
            logger.info("Report saved to: %s", args.output)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: trufflehog_check.py --path . [--entropy] [--output report.txt]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
