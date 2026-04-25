"""Version bumper utility for Flutter pubspec.yaml files.

This module provides functionality to read, bump, and write version numbers
in pubspec.yaml files. Supports major, minor, patch, and build level bumps,
with optional git tagging and pushing.
"""

import argparse
import logging
import re
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

# =============================================================================
# Constants
# =============================================================================

DEFAULT_PUBSPEC_PATH = "pubspec.yaml"
VERSION_PATTERN = re.compile(r"^version:\s*([^\s]+)", re.MULTILINE)
SEMVER_BUILD_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?")

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
# =============================================================================


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
        match = SEMVER_BUILD_PATTERN.match(version_str)
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


class BumpResult(NamedTuple):
    """Result of a version bump operation."""

    original: PubspecVersion
    bumped: PubspecVersion
    level: BumpLevel
    file_path: Path


# =============================================================================
# Core Functions
# =============================================================================


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
    match = VERSION_PATTERN.search(content)

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

    new_content, count = VERSION_PATTERN.subn(new_version_string, content)

    if count == 0:
        logger.error("Could not find version line to replace in %s", file_path)
        raise ValueError("No version line found in pubspec file")

    file_path.write_text(new_content, encoding="utf-8")
    logger.info("Updated version to %s in %s", version, file_path)


def tag_version(version: PubspecVersion, push: bool = False) -> None:
    """Create a git tag for the version and optionally push it."""
    tag_name = f"v{version}"
    logger.info("Creating git tag: %s", tag_name)

    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Version {tag_name}"],
            check=True,
            capture_output=True,
        )
        if push:
            logger.info("Pushing tag to origin: %s", tag_name)
            subprocess.run(["git", "push", "origin", tag_name], check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        logger.error("Git command failed: %s", err.stderr.decode().strip())
        raise


def bump_version(
    file_path: Path,
    level: BumpLevel,
    dry_run: bool = False,
    tag: bool = False,
    push: bool = False,
) -> BumpResult:
    """Bump version in pubspec.yaml file and optionally tag."""
    current = read_version_from_pubspec(file_path)
    bumped = calculate_next_version(current, level)

    if not dry_run:
        write_version_to_pubspec(file_path, bumped)
        if tag:
            tag_version(bumped, push=push)

    return BumpResult(
        original=current,
        bumped=bumped,
        level=level,
        file_path=file_path,
    )


# =============================================================================
# CLI
# =============================================================================


class BumpCommand:
    """CLI command for bumping version."""

    @staticmethod
    def configure_parser(parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for bump command."""
        parser.add_argument(
            "level",
            type=str,
            choices=["build", "patch", "minor", "major"],
            help="Bump level: build, patch, minor, or major",
        )
        parser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path(DEFAULT_PUBSPEC_PATH),
            help="Path to pubspec.yaml (default: %(default)s)",
        )
        parser.add_argument(
            "-d",
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes",
        )
        parser.add_argument(
            "-t",
            "--tag",
            action="store_true",
            help="Create a git tag after bumping",
        )
        parser.add_argument(
            "-p",
            "--push",
            action="store_true",
            help="Push the created git tag to origin",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> int:
        """Execute bump command."""
        level_map = {
            "build": BumpLevel.BUILD,
            "patch": BumpLevel.PATCH,
            "minor": BumpLevel.MINOR,
            "major": BumpLevel.MAJOR,
        }

        level = level_map[args.level]

        try:
            result = bump_version(
                file_path=args.file,
                level=level,
                dry_run=args.dry_run,
                tag=args.tag,
                push=args.push,
            )

            prefix = "[DRY RUN] " if args.dry_run else ""
            logger.info(
                "%sBumped %s: %s -> %s",
                prefix,
                result.level.name.lower(),
                result.original,
                result.bumped,
            )
            return 0

        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as err:
            logger.error("Failed to bump version: %s", err)
            return 1


def example_usage() -> None:
    """Demonstrate module functionality."""
    logger.info("Example: python bump_pubspec_version.py bump patch --tag --push")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Bump pubspec.yaml version")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    bump_parser = subparsers.add_parser("bump", help="Bump version")
    BumpCommand.configure_parser(bump_parser)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.command == "bump":
        sys.exit(BumpCommand.run(args))
    else:
        example_usage()


if __name__ == "__main__":
    main()
