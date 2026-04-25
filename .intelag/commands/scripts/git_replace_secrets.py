"""Rewrite git history with git-filter-repo to replace secret text or use a replace file."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "git-filter-repo"
REPLACEMENT_PLACEHOLDER = "***REMOVED***"
TEMP_REPLACE_FILE = "temp_replacements.txt"
REPLACE_FORMAT = "%s ==> %s"
FORCE_PUSH_CMD = ["git", "push", "origin", "--force", "--all"]


def _find_filter_repo_exe() -> Optional[str]:
    """Locate git-filter-repo executable or return None."""
    exe = shutil.which(TOOL_NAME)
    if exe:
        return exe
    scripts_dir = Path(sys.executable).parent
    suffix = "git-filter-repo.exe" if sys.platform == "win32" else "git-filter-repo"
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
    """Run git-filter-repo with --replace-text; optionally force push."""
    parser = argparse.ArgumentParser(description="Use git-filter-repo to rewrite history.")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to git repo",
    )
    parser.add_argument(
        "--replace-file",
        help="Path to replacements.txt file",
    )
    parser.add_argument(
        "--secret-text",
        help="A single secret string to remove (replaced with placeholder).",
    )
    parser.add_argument(
        "--force-push",
        action="store_true",
        help="Force push after rewriting",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm execution (required for non-interactive mode)",
    )
    args = parser.parse_args()

    filter_repo_exe = _find_filter_repo_exe()
    if not filter_repo_exe:
        logger.info("%s not found. Attempting to install...", TOOL_NAME)
        _install_tool(TOOL_NAME)
        filter_repo_exe = _find_filter_repo_exe()

    if filter_repo_exe:
        filter_repo_cmd: List[str] = [filter_repo_exe]
    else:
        filter_repo_cmd = [sys.executable, "-m", "git_filter_repo"]

    temp_file: Optional[str] = None
    replace_file_path = args.replace_file

    if args.secret_text:
        if replace_file_path:
            logger.warning("Both --replace-file and --secret-text provided; using --secret-text.")
        temp_file = TEMP_REPLACE_FILE
        with Path(temp_file).open("w", encoding="utf-8") as f:
            f.write(REPLACE_FORMAT % (args.secret_text, REPLACEMENT_PLACEHOLDER))
        replace_file_path = temp_file

    if not replace_file_path:
        logger.error("Must provide --replace-file or --secret-text.")
        sys.exit(1)

    cmd = [*filter_repo_cmd, "--replace-text", replace_file_path, "--force"]

    logger.info("Running: %s", " ".join(cmd))
    logger.warning("This will rewrite git history. Ensure you have backed up your repository.")

    if not args.confirm:
        logger.warning("SAFETY STOP: Provide --confirm to execute this destructive operation.")
        if temp_file and Path(temp_file).exists():
            Path(temp_file).unlink()
        sys.exit(1)

    try:
        result = subprocess.run(
            cmd,
            cwd=args.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.info("%s", result.stdout)
        if result.stderr:
            logger.warning("Stderr: %s", result.stderr)

        if result.returncode == 0:
            logger.info("History rewritten successfully.")
            logger.info("To push: git push origin --force --all")
            if args.force_push:
                logger.info("Attempting force push...")
                subprocess.run(
                    FORCE_PUSH_CMD,
                    cwd=args.repo_path,
                    check=True,
                )
                logger.info("Force push complete.")
        else:
            logger.error(
                "Rewriting failed with exit code %s",
                result.returncode,
            )
            sys.exit(result.returncode)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)
    finally:
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
            except OSError:
                pass


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: git_replace_secrets.py --repo-path . --secret-text SECRET --confirm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
