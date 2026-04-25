"""
╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-17 10:32:11
╰────────────────────────────────────────────────────────────────────────────────────────╯

Command-line interface for the type annotation fixer.
"""

import argparse
import logging

try:
    from modules.config import TypeAnnotationFixerConfig

    from type_annotation_fixer import TypeAnnotationFixer
except ImportError:
    from type_annotation_fixer.type_annotation_fixer import TypeAnnotationFixer

    from .config import TypeAnnotationFixerConfig


def main() -> int:
    """Command line interface for the type annotation fixer."""
    parser = argparse.ArgumentParser(
        description="Automatically fix missing type annotations using mypy analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════════════╗
                               📋 EXAMPLES
╠══════════════════════════════════════════════════════════════════════════════╣
 Basic type annotation fixing with dry run:
   type-annotation-fixer /path/to/project --dry-run

 Fix annotations with custom exclusions:
   type-annotation-fixer /path/to/project --exclude tests --exclude docs

 Fix annotations with custom mypy config:
   type-annotation-fixer /path/to/project --mypy-config mypy.ini

 Fix annotations without backups:
   type-annotation-fixer /path/to/project --no-backup

 Verbose fixing with all options:
   type-annotation-fixer /path/to/project --exclude build --exclude dist --verbose
╚══════════════════════════════════════════════════════════════════════════════╝

📖 CONFIGURATION GROUPS:
  Directory Processing - Specify target directory and exclusions
  Output Control       - Backup and dry-run options
  Analysis Settings    - Mypy configuration and verbosity
  Display Options      - Verbose output and progress information

🎯 QUICK START:
  1. Run dry-run first: type-annotation-fixer /path/to/project --dry-run
  2. Review changes and exclude directories if needed: --exclude <dir>
  3. Run actual fixing: type-annotation-fixer /path/to/project
  4. Use verbose mode for detailed progress: --verbose

🔐 SECURITY NOTES:
  • Always run --dry-run first to review changes
  • Backup files are created by default (.bak extension)
  • Use --exclude to avoid processing sensitive directories
  • Review added type annotations for correctness
  • Ensure mypy configuration is trusted and secure

This tool will:
  1. Run mypy analysis on each Python file
  2. Parse mypy output to find missing type annotations
  3. Automatically add basic type annotations based on inference
  4. Create backups before making changes (unless --no-backup)
""",
    )
    parser.add_argument("directory", help="Directory to process recursively")
    parser.add_argument(
        "--exclude",
        "-e",
        action="append",
        dest="exclude_dirs",
        metavar="DIR",
        help="Additional directories to exclude (can be used multiple times)",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Don't create backup files (.bak)"
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--mypy-config", "-c", metavar="CONFIG", help="Path to mypy configuration file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )
    parser.add_argument(
        "--autotyping", action="store_true", help="Enable autotyping for obvious annotations"
    )
    parser.add_argument(
        "--auto-annotate", action="store_true", help="Enable auto-type-annotate (Sentry)"
    )
    parser.add_argument(
        "--monkeytype", action="store_true", help="Enable MonkeyType type recording"
    )
    parser.add_argument(
        "--pytype", action="store_true", help="Enable Google pytype inference"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # Clean output for CLI
    )

    # Handle exclude directories safely
    exclude_dirs = None
    if args.exclude_dirs:
        exclude_dirs = set(args.exclude_dirs)

    try:
        config = TypeAnnotationFixerConfig(
            exclude_dirs=exclude_dirs,
            backup=not args.no_backup,
            dry_run=args.dry_run,
            mypy_config=args.mypy_config,
            enable_autotyping=args.autotyping,
            enable_auto_type_annotate=args.auto_annotate,
            enable_monkeytype=args.monkeytype,
            enable_pytype=args.pytype,
        )
        fixer = TypeAnnotationFixer(config=config)

        fixer.process_directory(args.directory)

    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTo install mypy: pip install mypy")
        return 1
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def example_usage() -> None:
    """Example usage of the type annotation fixer."""
    logging.basicConfig(level=logging.INFO)
    config = TypeAnnotationFixerConfig(dry_run=True)
    fixer = TypeAnnotationFixer(config=config)

    fixer.process_directory("intelag_pkg_manager")
