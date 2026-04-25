"""List large files from git-filter-repo path-all-sizes analysis output."""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ANALYSIS_FILE = Path(".git") / "filter-repo" / "analysis" / "path-all-sizes.txt"
DEFAULT_MIN_SIZE_MB = 10.0
BYTES_PER_MB = 1024 * 1024
REPORT_HEADER = "--- Git Large Files Report (threshold: %s MB) ---\n"


def main() -> None:
    """Read analysis file and print/write paths above the size threshold."""
    parser = argparse.ArgumentParser(
        description="List large files from git-filter-repo analysis."
    )
    parser.add_argument(
        "--size",
        type=float,
        default=DEFAULT_MIN_SIZE_MB,
        help="Minimum size in MB",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path for the report",
    )
    args = parser.parse_args()

    if not ANALYSIS_FILE.exists():
        logger.error("Analysis file not found at %s", ANALYSIS_FILE)
        analysis_dir = ANALYSIS_FILE.parent
        if analysis_dir.exists():
            for name in analysis_dir.iterdir():
                logger.info("  %s", name)
        else:
            logger.error("Directory %s does not exist.", analysis_dir)
        logger.info("Run 'git filter-repo --analyze' first.")
        sys.exit(1)

    logger.info("Files larger than %s MB", args.size)
    found = False
    with ANALYSIS_FILE.open(encoding="utf-8") as analysis_file:
        lines = analysis_file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        if "Size" in line and "Path" in line:
            continue

        parts = line.split(None, 2)
        if len(parts) < 3:
            continue

        try:
            size_bytes = int(parts[0])
            path = parts[2]
        except ValueError:
            continue

        size_mb = size_bytes / BYTES_PER_MB
        if size_mb >= args.size:
            result_line = f"{size_mb:.2f} MB | {path}"
            logger.info("%s", result_line)
            if args.output:
                with Path(args.output).open("a", encoding="utf-8") as out:
                    out.write(result_line + "\n")
            found = True

    if not found:
        msg = "No files found exceeding the size limit."
        logger.info("%s", msg)
        if args.output:
            with Path(args.output).open("a", encoding="utf-8") as out:
                out.write(msg + "\n")
    elif args.output:
        logger.info("Report saved to: %s", args.output)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info(
        "Example: git_list_large_files.py --size 10 --output .intelag/reports/large.txt"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--output" in sys.argv:
        idx = sys.argv.index("--output") + 1
        if idx < len(sys.argv):
            out_path = sys.argv[idx]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            size_val = (
                sys.argv[sys.argv.index("--size") + 1]
                if "--size" in sys.argv
                else str(DEFAULT_MIN_SIZE_MB)
            )
            with Path(out_path).open("w", encoding="utf-8") as report_file:
                report_file.write(REPORT_HEADER % size_val)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
