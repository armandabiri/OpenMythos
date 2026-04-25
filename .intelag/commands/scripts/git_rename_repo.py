"""
Rename the current Git repo: rename on the cloud (e.g. GitHub via gh CLI) and update the local remote URL.

Run from the repository root. If GitHub CLI (gh) is installed and authenticated,
renames the repo on GitHub; otherwise only updates the local remote URL
(you must rename the repo manually in the host's UI).
"""

import argparse
import logging
import re
import subprocess
import sys
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_REMOTE = "origin"
GITHUB_HOSTS = ("github.com", "www.github.com")


def run_git(args: list[str], check: bool = True) -> str:
    """Run git and return stdout; exit on failure if check."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    out = (result.stdout or "").strip()
    if check and result.returncode != 0:
        logger.error("git %s failed: %s", " ".join(args), result.stderr or out)
        sys.exit(result.returncode)
    return out


def get_remote_url(remote: str) -> str:
    """Return the URL of the given remote."""
    return run_git(["remote", "get-url", remote])


def parse_remote_url(url: str) -> tuple[str, str, str]:
    """Return (host, owner_or_org, repo_name). Repo name without .git."""
    url = url.strip()
    if url.startswith("git@"):
        # git@github.com:owner/repo.git
        match = re.match(r"git@([^:]+):([^/]+)/([^/]+?)(\.git)?$", url)
        if match:
            host, owner, repo = match.group(1), match.group(2), match.group(3)
            return host, owner, repo
    else:
        # https://github.com/owner/repo.git or https://host/owner/repo
        parsed = urlparse(url)
        host = parsed.hostname or ""
        path = (parsed.path or "").strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].removesuffix(".git") if parts[1].endswith(".git") else parts[1]
            return host, owner, repo
    logger.error("Could not parse remote URL: %s", url)
    sys.exit(1)


def build_new_url(host: str, owner: str, new_repo_name: str, current_url: str) -> str:
    """Build new remote URL with new repo name; preserve protocol (https vs git@)."""
    if current_url.strip().startswith("git@"):
        return f"git@{host}:{owner}/{new_repo_name}.git"
    return f"https://{host}/{owner}/{new_repo_name}.git"


def rename_on_github(new_name: str) -> bool:
    """Rename the repo on GitHub using gh CLI. Return True if success."""
    try:
        result = subprocess.run(
            ["gh", "repo", "rename", new_name, "--yes"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Renamed repo on GitHub to: %s", new_name)
            return True
        logger.warning(
            "gh repo rename failed (not authenticated or not GitHub?): %s",
            (result.stderr or result.stdout or "").strip(),
        )
    except FileNotFoundError:
        logger.warning("GitHub CLI (gh) not found. Install it or rename the repo manually in GitHub settings.")
    return False


def main() -> None:
    """Rename repo on cloud (if GitHub + gh) and update local remote URL."""
    parser = argparse.ArgumentParser(description="Rename the repo in the cloud and update the local remote URL.")
    parser.add_argument(
        "--new-name",
        required=True,
        help="New repository name (e.g. my-project-v2).",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="Remote to update (default: origin).",
    )
    parser.add_argument(
        "--skip-cloud",
        action="store_true",
        help=("Only update local remote URL; do not rename on GitHub (use if you already renamed in the UI)."),
    )
    args = parser.parse_args()

    new_name = args.new_name.strip()
    if not new_name:
        logger.error("--new-name must be non-empty.")
        sys.exit(1)

    current_url = get_remote_url(args.remote)
    host, owner, old_repo = parse_remote_url(current_url)

    if old_repo == new_name:
        logger.info("Remote already points to name '%s'. Nothing to do.", new_name)
        return

    if not args.skip_cloud:
        is_github = any(h in host for h in GITHUB_HOSTS) or host == "github.com"
        if is_github:
            rename_on_github(new_name)
        else:
            logger.info(
                "Not a GitHub host (%s). Rename the repo manually in your host's UI, "
                "then run again with --skip-cloud to update the remote URL only.",
                host,
            )
            logger.info("Or run with --skip-cloud now to only update the local remote URL.")

    new_url = build_new_url(host, owner, new_name, current_url)
    run_git(["remote", "set-url", args.remote, new_url])
    logger.info("Updated remote '%s' to: %s", args.remote, new_url)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: git_rename_repo.py --new-name my-repo-v2 [--remote origin] [--skip-cloud]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
