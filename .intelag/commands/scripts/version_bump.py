"""Version bumper utility for Python TOML project files.

This module provides functionality to read, bump, and write version numbers
in pyproject.toml files. Supports semantic versioning with major, minor,
and patch level bumps with automatic overflow handling.
"""

import argparse
import logging
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOML_PATH = "pyproject.toml"
DEFAULT_VERSION = "0.0.0"
VERSION_PATTERN = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
MAX_VERSION_SEGMENT = 99


# =============================================================================
# Enums
# =============================================================================


class BumpLevel(IntEnum):
    """Version bump levels for semantic versioning."""

    PATCH = 1
    MINOR = 2
    MAJOR = 3


# =============================================================================
# Data Classes
# =============================================================================


class Version(NamedTuple):
    """Semantic version representation."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Return version as string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Parse version string into Version object."""
        parts = version_str.strip().split(".")

        major = _parse_version_segment(parts, 0)
        minor = _parse_version_segment(parts, 1)
        patch = _parse_version_segment(parts, 2)

        return cls(major=major, minor=minor, patch=patch)


class BumpResult(NamedTuple):
    """Result of a version bump operation."""

    original: Version
    bumped: Version
    level: BumpLevel
    file_path: Path


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_version_segment(parts: list[str], index: int) -> int:
    """Parse a single version segment safely."""
    if index >= len(parts):
        return 0

    segment = parts[index]
    if segment.isdigit():
        return int(segment)

    return 0


def _apply_overflow(value: int, next_value: int) -> tuple[int, int]:
    """Apply overflow logic when version segment exceeds maximum."""
    if value > MAX_VERSION_SEGMENT:
        return 0, next_value + 1
    return value, next_value


# =============================================================================
# Core Functions
# =============================================================================


def calculate_next_version(current: Version, level: BumpLevel) -> Version:
    """Calculate next version based on bump level."""
    major = current.major
    minor = current.minor
    patch = current.patch

    if level == BumpLevel.PATCH:
        patch += 1
        patch, minor = _apply_overflow(patch, minor)
        minor, major = _apply_overflow(minor, major)

    elif level == BumpLevel.MINOR:
        minor += 1
        patch = 0
        minor, major = _apply_overflow(minor, major)

    elif level == BumpLevel.MAJOR:
        major += 1
        minor = 0
        patch = 0

    return Version(major=major, minor=minor, patch=patch)


def read_version_from_toml(file_path: Path) -> Version:
    """Read current version from TOML file."""
    if not file_path.exists():
        logger.error("TOML file not found: %s", file_path)
        raise FileNotFoundError(f"TOML file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)

    if not match:
        logger.warning("No version found in %s, using default", file_path)
        return Version.from_string(DEFAULT_VERSION)

    version_str = match.group(1)
    logger.debug("Found version %s in %s", version_str, file_path)

    return Version.from_string(version_str)


def write_version_to_toml(file_path: Path, version: Version) -> None:
    """Write version to TOML file."""
    if not file_path.exists():
        logger.error("TOML file not found: %s", file_path)
        raise FileNotFoundError(f"TOML file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    new_version_line = f'version = "{version}"'

    new_content, count = VERSION_PATTERN.subn(new_version_line, content)

    if count == 0:
        logger.error("Could not find version line to replace in %s", file_path)
        raise ValueError("No version line found in TOML file")

    file_path.write_text(new_content, encoding="utf-8")
    logger.info("Updated version to %s in %s", version, file_path)


def bump_version(
    file_path: Path,
    level: BumpLevel,
    dry_run: bool = False,
) -> BumpResult:
    """Bump version in TOML file."""
    current = read_version_from_toml(file_path)
    bumped = calculate_next_version(current, level)

    logger.info(
        "Bumping %s: %s -> %s",
        level.name.lower(),
        current,
        bumped,
    )

    if not dry_run:
        write_version_to_toml(file_path, bumped)

    return BumpResult(
        original=current,
        bumped=bumped,
        level=level,
        file_path=file_path,
    )


def revert_version(file_path: Path, original: Version) -> None:
    """Revert version to original value."""
    write_version_to_toml(file_path, original)
    logger.info("Reverted version to %s", original)


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
            choices=["patch", "minor", "major", "1", "2", "3"],
            help="Bump level: patch/1, minor/2, or major/3",
        )
        parser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path(DEFAULT_TOML_PATH),
            help="Path to pyproject.toml (default: %(default)s)",
        )
        parser.add_argument(
            "-d",
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> int:
        """Execute bump command."""
        level_map: dict[str, BumpLevel] = {
            "patch": BumpLevel.PATCH,
            "minor": BumpLevel.MINOR,
            "major": BumpLevel.MAJOR,
            "1": BumpLevel.PATCH,
            "2": BumpLevel.MINOR,
            "3": BumpLevel.MAJOR,
        }

        level = level_map[args.level]

        try:
            result = bump_version(
                file_path=args.file,
                level=level,
                dry_run=args.dry_run,
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

        except FileNotFoundError:
            logger.error("File not found: %s", args.file)
            return 1

        except ValueError as err:
            logger.error("Version error: %s", err)
            return 1


class ShowCommand:
    """CLI command for showing current version."""

    @staticmethod
    def configure_parser(parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for show command."""
        parser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path(DEFAULT_TOML_PATH),
            help="Path to pyproject.toml (default: %(default)s)",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> int:
        """Execute show command."""
        try:
            version = read_version_from_toml(args.file)
            logger.info("Current version: %s", version)
            return 0

        except FileNotFoundError:
            logger.error("File not found: %s", args.file)
            return 1


class SetCommand:
    """CLI command for setting specific version."""

    @staticmethod
    def configure_parser(parser: argparse.ArgumentParser) -> None:
        """Configure argument parser for set command."""
        parser.add_argument(
            "version",
            type=str,
            help="Version to set (e.g., 1.2.3)",
        )
        parser.add_argument(
            "-f",
            "--file",
            type=Path,
            default=Path(DEFAULT_TOML_PATH),
            help="Path to pyproject.toml (default: %(default)s)",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> int:
        """Execute set command."""
        try:
            version = Version.from_string(args.version)
            write_version_to_toml(args.file, version)
            logger.info("Set version to %s", version)
            return 0

        except FileNotFoundError:
            logger.error("File not found: %s", args.file)
            return 1

        except ValueError as err:
            logger.error("Invalid version: %s", err)
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="version_bumper",
        description="Version bumper for Python TOML project files",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # Bump command
    bump_parser = subparsers.add_parser(
        "bump",
        help="Bump version by level",
    )
    BumpCommand.configure_parser(bump_parser)

    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show current version",
    )
    ShowCommand.configure_parser(show_parser)

    # Set command
    set_parser = subparsers.add_parser(
        "set",
        help="Set specific version",
    )
    SetCommand.configure_parser(set_parser)

    return parser


def configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def cmd_main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    command_map: dict[str, type[BumpCommand] | type[ShowCommand] | type[SetCommand]] = {
        "bump": BumpCommand,
        "show": ShowCommand,
        "set": SetCommand,
    }

    command_class = command_map.get(args.command)
    if command_class is None:
        logger.error("Unknown command: %s", args.command)
        return 1

    return command_class.run(args)


# =============================================================================
# Example Usage
# =============================================================================


def example_usage() -> None:
    """Demonstrate module functionality."""
    configure_logging(verbose=True)

    # Create a test TOML file
    test_file = Path("test_pyproject.toml")
    test_content = """[project]
name = "test-package"
version = "1.2.3"
description = "A test package"
"""
    test_file.write_text(test_content, encoding="utf-8")

    try:
        # Show current version
        version = read_version_from_toml(test_file)
        logger.info("Current version: %s", version)

        # Bump patch
        result = bump_version(test_file, BumpLevel.PATCH)
        logger.info("After patch bump: %s", result.bumped)

        # Bump minor
        result = bump_version(test_file, BumpLevel.MINOR)
        logger.info("After minor bump: %s", result.bumped)

        # Bump major
        result = bump_version(test_file, BumpLevel.MAJOR)
        logger.info("After major bump: %s", result.bumped)

        # Test overflow
        overflow_version = Version(major=1, minor=99, patch=99)
        write_version_to_toml(test_file, overflow_version)
        result = bump_version(test_file, BumpLevel.PATCH)
        logger.info("After overflow bump: %s", result.bumped)

        # Revert
        revert_version(test_file, result.original)
        logger.info("After revert: %s", read_version_from_toml(test_file))

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
            logger.info("Cleaned up test file")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cmd_main())
    else:
        example_usage()
