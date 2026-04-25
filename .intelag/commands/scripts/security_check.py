"""Run GitGuardian ggshield secret scan on a path or full repo history."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "ggshield"
REPORT_SEP = "-" * 40
AUTH_HINT = "Please run: ggshield auth login\nThen run this scan again."


def _find_ggshield_exe() -> Optional[str]:
    """Locate ggshield executable or return None."""
    exe = shutil.which(TOOL_NAME)
    if exe:
        return exe
    scripts_dir = Path(sys.executable).parent
    suffix = "ggshield.exe" if sys.platform == "win32" else "ggshield"
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
    """Run ggshield secret scan (path or repo); optionally write report to file."""
    parser = argparse.ArgumentParser(description="Run security checks using ggshield.")
    parser.add_argument(
        "--path",
        default=".",
        help="Path to scan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for the report",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Scan full git history (repo scan)",
    )
    args = parser.parse_args()

    ggshield_exe = _find_ggshield_exe()
    if not ggshield_exe:
        logger.info("%s not found. Attempting to install...", TOOL_NAME)
        _install_tool(TOOL_NAME)
        ggshield_exe = _find_ggshield_exe()
    if not ggshield_exe:
        ggshield_exe = TOOL_NAME

    if args.full_history:
        cmd: List[str] = [ggshield_exe, "secret", "scan", "repo", args.path]
    else:
        cmd = [ggshield_exe, "secret", "scan", "path", "-r", args.path]

    logger.info("Running: %s", " ".join(cmd))
    logger.info("Note: Requires GitGuardian API Key. If not logged in, run: ggshield auth login")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""

        logger.info("%s", stdout)
        if stderr:
            logger.warning("Stderr: %s", stderr)

        if result.returncode != 0:
            if "auth login" in stderr or "auth login" in stdout or "API key" in stderr:
                logger.warning("Authentication required. %s", AUTH_HINT)
                sys.exit(127) # Specialized "auth missing" code
            elif result.returncode == 1:
                logger.warning("Secrets found. Check the report above.")
            else:
                logger.error("Scan failed with exit code %s", result.returncode)

        if args.output:
            output_path = Path(args.output)
            out_dir = output_path.parent
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(REPORT_SEP + "\n\n")
                f.write(stdout)
                f.write("\n" + REPORT_SEP + "\n")
                f.write(stderr)
            logger.info("Report saved to: %s", args.output)

        sys.exit(result.returncode)
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: security_check.py --path . [--full-history] [--output report.txt]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
