"""Build utility for Flutter projects with version bumping and tagging support.

This module provides functionality to bump the version in pubspec.yaml,
run flutter build for various targets, and optionally create/push git tags.
"""

import argparse
import logging
import re
import shutil
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

# =============================================================================
# Constants
# =============================================================================

DEFAULT_PUBSPEC_PATH = "pubspec.yaml"
VERSION_PATTERN_STR = r"^version:\s*([^\s]+)"
SEMVER_BUILD_PATTERN_STR = r"(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?"

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================


class BumpLevel(IntEnum):
    """Version bump levels for Flutter pubspec versioning."""

    BUILD = 1
    PATCH = 2
    MINOR = 3
    MAJOR = 4


# =============================================================================
# Data Classes
# ================= ============================================================


class PubspecVersion(NamedTuple):
    """Flutter version representation (major.minor.patch+build)."""

    major: int
    minor: int
    patch: int
    build: int

    def __str__(self) -> str:
        """Return version as string."""
        return f"{self.major}.{self.minor}.{self.patch}+{self.build}"

    @classmethod
    def from_string(cls, version_str: str) -> "PubspecVersion":
        """Parse version string into PubspecVersion object."""
        match = re.match(SEMVER_BUILD_PATTERN_STR, version_str)
        if not match:
            logger.error("Invalid version format: %s", version_str)
            raise ValueError(f"Invalid version format: {version_str}")

        major, minor, patch, build = match.groups()
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            build=int(build) if build else 0,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def run_command(command: list[str]) -> bool:
    """Run a shell command and stream output."""
    logger.info("Executing: %s", " ".join(command))
    try:
        # Resolve executable path to avoid shell dependencies
        if command:
            exe = shutil.which(command[0])
            if exe:
                command[0] = exe

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        if process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        process.wait()
        return process.returncode == 0
    except (subprocess.SubprocessError, OSError) as err:
        logger.error("Failed to execute command: %s", err)
        return False


def calculate_next_version(current: PubspecVersion, level: BumpLevel) -> PubspecVersion:
    """Calculate next version based on bump level."""
    major, minor, patch, build = current.major, current.minor, current.patch, current.build

    if level == BumpLevel.MAJOR:
        major += 1
        minor = 0
        patch = 0
        build += 1
    elif level == BumpLevel.MINOR:
        minor += 1
        patch = 0
        build += 1
    elif level == BumpLevel.PATCH:
        patch += 1
        build += 1
    elif level == BumpLevel.BUILD:
        build += 1

    return PubspecVersion(major=major, minor=minor, patch=patch, build=build)


def read_version_from_pubspec(file_path: Path) -> PubspecVersion:
    """Read current version from pubspec.yaml file."""
    if not file_path.exists():
        logger.error("pubspec file not found: %s", file_path)
        raise FileNotFoundError(f"pubspec file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    match = re.search(VERSION_PATTERN_STR, content, re.MULTILINE)

    if not match:
        logger.error("No version found in %s", file_path)
        raise ValueError(f"No version found in {file_path}")

    version_str = match.group(1)
    return PubspecVersion.from_string(version_str)


def write_version_to_pubspec(file_path: Path, version: PubspecVersion) -> None:
    """Write version to pubspec.yaml file."""
    if not file_path.exists():
        logger.error("pubspec file not found: %s", file_path)
        raise FileNotFoundError(f"pubspec file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    new_version_string = f"version: {version}"

    new_content, count = re.subn(VERSION_PATTERN_STR, new_version_string, content, flags=re.MULTILINE)

    if count == 0:
        logger.error("Could not find version line to replace in %s", file_path)
        raise ValueError("No version line found in pubspec file")

    file_path.write_text(new_content, encoding="utf-8")
    logger.info("Updated version to %s in %s", version, file_path)


def tag_version(version: PubspecVersion, push: bool = False) -> bool:
    """Create a git tag for the version and optionally push it."""
    tag_name = f"v{version}"
    logger.info("Creating git tag: %s", tag_name)

    tag_cmd = ["git", "tag", "-a", tag_name, "-m", f"Version {tag_name}"]
    if run_command(tag_cmd):
        if push:
            logger.info("Pushing tag to origin: %s", tag_name)
            return run_command(["git", "push", "origin", tag_name])
        return True
    return False


# =============================================================================
# CLI
# =============================================================================


def example_usage() -> None:
    """Print example usage instructions."""
    logger.info("Example: python build.py --part patch --target windows --tag")


def cmd_main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Build script with version bumping for Flutter.")
    parser.add_argument(
        "--part",
        choices=["major", "minor", "patch", "build"],
        default="build",
        help="Which part of the version to bump",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without changing files")
    parser.add_argument("--build-only", action="store_true", help="Skip version bumping and only run build")
    parser.add_argument("--skip-build", action="store_true", help="Skip the flutter build step and only bump version")
    parser.add_argument("--target", default="windows", help="Build target (windows, apk, ios, etc.)")
    parser.add_argument("-t", "--tag", action="store_true", help="Create a git tag after bumping")
    parser.add_argument("-p", "--push", action="store_true", help="Push the created git tag to origin")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    pubspec_path = Path(DEFAULT_PUBSPEC_PATH)

    # 1. Bump version
    if not args.build_only:
        try:
            current = read_version_from_pubspec(pubspec_path)
            level_map = {
                "build": BumpLevel.BUILD,
                "patch": BumpLevel.PATCH,
                "minor": BumpLevel.MINOR,
                "major": BumpLevel.MAJOR,
            }
            new_version = calculate_next_version(current, level_map[args.part])
            logger.info("New version: %s", new_version)

            if not args.dry_run:
                write_version_to_pubspec(pubspec_path, new_version)
                if args.tag:
                    if not tag_version(new_version, push=args.push):
                        logger.error("Failed to tag version")
                        return 1
        except (FileNotFoundError, ValueError) as err:
            logger.error("Versioning failed: %s", err)
            return 1

    if args.dry_run:
        logger.info("Dry run complete. No changes made.")
        return 0

    # 2. Run Flutter build
    if not args.skip_build:
        logger.info("Starting build for %s...", args.target)
        success = run_command(["flutter", "build", args.target])

        if success:
            logger.info("Build completed successfully!")
        else:
            logger.error("Build failed!")
            return 1
    else:
        logger.info("Skipping build step as requested.")

    return 0


if __name__ == "__main__":
    sys.exit(cmd_main())
