"""Remove specified paths from git history using git-filter-repo."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

LOG_HEADER = "--- Git Removal Log ---\n"
SUCCESS_MSG = (
    "\nSuccessfully removed files from history.\n"
    "Note: You may need to run 'git push origin --force --all' to update remote.\n"
)


def main() -> None:
    """Build git_filter_repo command and run it; optionally append to output log."""
    parser = argparse.ArgumentParser(description="Remove files from git history using git-filter-repo.")
    parser.add_argument(
        "--files",
        type=str,
        required=True,
        help="Space or newline separated list of files to remove",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path for the log",
    )
    args = parser.parse_args()

    files_to_remove: List[str] = [f.strip() for f in args.files.replace("\n", " ").split(" ") if f.strip()]

    if not files_to_remove:
        logger.error("No files specified.")
        sys.exit(1)

    logger.info("Preparing to remove %d paths from history...", len(files_to_remove))

    cmd: List[str] = [sys.executable, "-m", "git_filter_repo"]
    for file_path in files_to_remove:
        cmd.extend(["--path", file_path])
    cmd.extend(["--invert-paths", "--force"])

    logger.info("Running: %s", " ".join(cmd))

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            for line in process.stdout:
                logger.info("%s", line.rstrip())
        process.wait()
        if process.returncode == 0:
            logger.info("%s", SUCCESS_MSG.strip())
            if args.output:
                with Path(args.output).open("a", encoding="utf-8") as output_log_file:
                    output_log_file.write(f"\nCommand: {' '.join(cmd)}\n")
                    output_log_file.write(SUCCESS_MSG)
                    output_log_file.write(f"Removed files: {', '.join(files_to_remove)}\n")
        else:
            logger.error(
                "git filter-repo failed with return code %s",
                process.returncode,
            )
            sys.exit(process.returncode)
    except (subprocess.SubprocessError, OSError) as e:
        logger.exception("Error executing git filter-repo: %s", e)
        sys.exit(1)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: remove_files.py --files 'path/a path/b' [--output log.txt]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            Path(sys.argv[idx]).parent.mkdir(parents=True, exist_ok=True)
            with Path(sys.argv[idx]).open("w", encoding="utf-8") as header_file:
                header_file.write(LOG_HEADER)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
