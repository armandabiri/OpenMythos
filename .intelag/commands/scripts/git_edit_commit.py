"""Run 'git rebase -i <commit>^' with edit on that commit, then 'git reset HEAD^'."""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STASH_MESSAGE = "Edit Past Commit: auto-stash"
HINT_STASH = 'Stash or commit your changes first (e.g. git stash push -u -m "WIP"), then run again.'
EDITOR_SCRIPT_BODY = (
    "import sys\n"
    "p = sys.argv[1]\n"
    "c = open(p, encoding='utf-8').read()\n"
    "open(p, 'w', encoding='utf-8').write(c.replace('pick', 'edit', 1))\n"
)


def run_git(
    args: list[str],
    env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> Optional[str]:
    """Run git with given args (no shell) so refs like commit^ work on Windows."""
    env_map = env if env is not None else dict(os.environ)
    cmd = ["git", *args]
    try:
        result = subprocess.run(
            cmd,
            env=env_map,
            check=check,
            capture_output=True,
            text=True,
        )
        return (result.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        logger.error("Command failed: %s", " ".join(cmd))
        if stdout:
            logger.error("%s", stdout)
        if stderr:
            logger.error("%s", stderr)
        if "unstaged changes" in stderr.lower() or "uncommitted changes" in stderr.lower():
            logger.info("%s", HINT_STASH)
        if check:
            sys.exit(e.returncode)
        return None


def main() -> None:
    """Uncommit a previous commit via rebase -i and reset HEAD^."""
    parser = argparse.ArgumentParser(description="Automate 'git rebase -i <commit>^' -> edit -> 'git reset HEAD^'")
    parser.add_argument(
        "commit",
        help="The commit hash or reference to edit",
    )
    args = parser.parse_args()

    commit = args.commit.strip()
    if not commit:
        logger.error("Commit hash is empty.")
        sys.exit(1)

    did_stash = False
    status = ""
    try:
        status = run_git(["status", "--porcelain"], check=False) or ""
    except Exception:
        pass
    if status and status.strip():
        r = subprocess.run(
            ["git", "stash", "push", "-u", "-m", STASH_MESSAGE],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            logger.error(
                "%s",
                (r.stderr or r.stdout or "Stash failed.").strip(),
            )
            sys.exit(1)
        did_stash = True

    rebase_ref = f"{commit}^"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as f:
        f.write(EDITOR_SCRIPT_BODY)
        editor_script = f.name

    try:
        env = os.environ.copy()
        env["GIT_SEQUENCE_EDITOR"] = f'"{sys.executable}" "{editor_script}"'

        logger.info("Starting rebase for commit %s...", commit)
        run_git(["rebase", "-i", rebase_ref], env=env)

        logger.info("Rebase stopped. Resetting HEAD^ to uncommit changes...")
        run_git(["reset", "HEAD^"])

        logger.info("Commit uncommitted; changes are in the working directory.")
        if did_stash:
            logger.info("Previous changes were auto-stashed. After rebase --continue, run: git stash pop")
        logger.info("After making changes: git add <files>, git commit --amend, git rebase --continue")
    finally:
        try:
            Path(editor_script).unlink()
        except OSError:
            pass


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: git_edit_commit.py HEAD~1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
