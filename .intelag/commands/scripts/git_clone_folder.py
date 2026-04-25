"""
Turn an existing folder into its own Git repo, push to GitHub, and optionally
add it back to the current project as a git submodule.

Steps:
1. In FOLDER: git init (if not already a repo), add all, commit, remote add, push.
2. If --add-as-submodule: from workspace root, run git submodule add <url> <folder>
   (Git subsumes the existing repo when the path is already a valid git repository).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

DEFAULT_COMMIT_MESSAGE = "Initial commit"
DEFAULT_BRANCH = "main"
REPLACEMENT_PLACEHOLDER = "***REMOVED***"
STASH_MESSAGE = "Edit Past Commit: auto-stash"


def run_git(
    args: list[str],
    cwd: str | None = None,
    check: bool = True,
    capture: bool = True,
) -> Tuple[str, str, int]:
    """Run git with given args; returns (stdout, stderr, returncode)."""
    cmd = ["git", *args]
    result = subprocess.run(
        cmd,
        cwd=cwd or str(Path.cwd()),
        capture_output=capture,
        text=True,
        check=False,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if check and result.returncode != 0:
        logger.error("Command failed: %s", " ".join(cmd))
        if out:
            logger.error("%s", out)
        if err:
            logger.error("%s", err)
        sys.exit(result.returncode)
    return out, err, result.returncode


def is_git_repo(path: str) -> bool:
    """Return True if path is the root of a git repository."""
    git_dir = Path(path) / ".git"
    return git_dir.is_dir() or git_dir.is_file()


def main() -> None:
    """Turn folder into repo, push to GitHub, optionally add as submodule."""
    parser = argparse.ArgumentParser(
        description="Turn a folder into its own repo, push to GitHub, optionally add as submodule."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the folder (relative to current dir or absolute).",
    )
    parser.add_argument(
        "--repo-url",
        required=False,
        help="GitHub repo URL (e.g. https://github.com/org/repo.git). Defaults to INTELAG organization if empty.",
    )
    parser.add_argument(
        "--add-as-submodule",
        action="store_true",
        help="After pushing, add the repo as a submodule in the current project.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Message for the initial commit.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Branch name to create and push (default: main).",
    )
    # Ignore unknown args (e.g. "[object Object]" from JS callers or PowerShell-split tokens)
    args, _ = parser.parse_known_args()

    workspace_root = Path.cwd()
    folder_path = (workspace_root / args.folder).resolve()
    folder_rel = folder_path.relative_to(workspace_root).as_posix()

    if not folder_path.is_dir():
        logger.error("Folder does not exist: %s", folder_path)
        sys.exit(1)

    if not args.repo_url:
        folder_name = folder_path.name
        args.repo_url = f"https://github.com/INTELAG/{folder_name}.git"
        logger.info("Using default repo URL: %s", args.repo_url)

    if not is_git_repo(folder_path):
        logger.info("Initializing git repo in %s...", folder_path)
        run_git(["init"], cwd=str(folder_path))
    else:
        logger.info("Folder is already a git repo.")

    _, _, remote_code = run_git(["remote", "get-url", "origin"], cwd=str(folder_path), check=False)
    if remote_code != 0:
        run_git(["remote", "add", "origin", args.repo_url], cwd=str(folder_path))

    run_git(["add", "."], cwd=str(folder_path))
    _, _, code = run_git(["diff", "--staged", "--quiet"], cwd=str(folder_path), check=False)
    if code != 0:
        run_git(["commit", "-m", args.commit_message], cwd=str(folder_path))
    else:
        _, _, code = run_git(["diff", "--quiet"], cwd=str(folder_path), check=False)
        if code != 0:
            run_git(["commit", "-m", args.commit_message], cwd=str(folder_path))

    run_git(["branch", "-M", args.branch], cwd=str(folder_path))
    logger.info("Pushing to %s...", args.repo_url)
    run_git(["push", "-u", "origin", args.branch], cwd=str(folder_path))
    logger.info("Pushed successfully.")

    if args.add_as_submodule:
        try:
            run_git(["rev-parse", "--show-toplevel"], cwd=str(workspace_root))
        except Exception:  # pylint: disable=broad-except
            logger.warning("Current directory is not inside a git repo; skipping submodule add.")
            return

        _, _, code = run_git(
            ["ls-files", "--error-unmatch", folder_rel],
            cwd=str(workspace_root),
            check=False,
        )
        if code == 0:
            logger.info(
                "Removing %s from parent index before adding as submodule...",
                folder_rel,
            )
            run_git(["rm", "-r", "--cached", folder_rel], cwd=str(workspace_root))

        logger.info("Adding submodule at %s...", folder_rel)
        run_git(
            ["submodule", "add", args.repo_url, folder_rel],
            cwd=str(workspace_root),
        )
        logger.info("Submodule added successfully.")


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: folder_to_repo.py --folder path/to/dir --repo-url https://github.com/org/repo.git")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
