"""Secure Python scripts using PyArmor (obfuscate and output to a directory)."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = ".intelag/commands/scripts"
DEFAULT_OUTPUT = ".intelag/commands/secure_scripts"
PYARMOR_TOOL = "pyarmor"


def main() -> None:
    """Install PyArmor if needed and run pyarmor gen to secure scripts."""
    parser = argparse.ArgumentParser(description="Secure Python scripts using PyArmor.")
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_PATH,
        help="Path to scripts directory to secure",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output directory for secured scripts",
    )
    args = parser.parse_args()

    if not Path(args.path).exists():
        logger.error("Path does not exist: %s", args.path)
        sys.exit(1)

    logger.info("Installing PyArmor...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", PYARMOR_TOOL])
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install pyarmor: %s", e)
        sys.exit(1)

    logger.info("Securing scripts in %s...", args.path)
    try:
        cmd = ["pyarmor", "gen", "-O", args.output, args.path]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        logger.info("Secured scripts are in: %s", args.output)
        logger.info("Output contains 'pyarmor_runtime' folder required to run the scripts.")
    except (subprocess.CalledProcessError, OSError) as e:
        logger.exception("Error during obfuscation: %s", e)
        sys.exit(1)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: secure_scripts.py --path .intelag/commands/scripts --output .intelag/commands/secure_scripts")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
