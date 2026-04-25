"""
  Log lazy formatter module.
  This module provides functionality to convert f-string and .format() logging calls
  to lazy % formatting for better performance and security in logging operations.
  It includes both a command-line tool and programmatic API for batch conversion
  of Python files in a directory structure.

╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-25 18:36:55
╰────────────────────────────────────────────────────────────────────────────────────────╯
"""

# Standard library imports
import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed  # pylint: disable=no-name-in-module
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

# Third-party imports
# None

# Internal imports
try:
    from models.config import LogLazyFormatterConfig
except ImportError:
    from .models.config import LogLazyFormatterConfig

# Module logger
module_logger = logging.getLogger(__name__)


class LogFormatConverter:
    """
    Convert f-string and .format() logging calls to lazy % formatting.
    """

    # Attribute declarations
    root_path: Optional[Path]

    def __init__(
        self,
        exclude_dirs: Optional[Set[str]] = None,
        backup: bool = True,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize the converter.

        Args:
            exclude_dirs: Additional directories to exclude
            backup: Whether to create backups before modification
            dry_run: If True, only show what would be changed
        """
        # Load configuration
        self.config = LogLazyFormatterConfig()

        self.exclude_dirs = self.config.default_exclude_dirs.copy()
        if exclude_dirs:
            self.exclude_dirs.update(exclude_dirs)
        self.backup = backup
        self.dry_run = dry_run
        self.changes_made = 0
        self.files_processed = 0
        self.root_path: Optional[Path] = None

    def find_python_files(self, root_dir: Path) -> List[Path]:
        """Recursively find Python files, excluding specified directories."""
        python_files: List[Path] = []

        for root, dirs, files in os.walk(root_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                if file.endswith(tuple(self.config.python_extensions)):
                    python_files.append(Path(root) / file)

        return python_files

    def extract_f_string_content(self, f_string: str) -> Tuple[str, List[str]]:
        """
        Extract format string and variables from f-string.

        Args:
            f_string: The f-string content (without f'' quotes)

        Returns:
            Tuple of (format_string, variable_list)
        """
        # Simple regex to find {expression} patterns
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, f_string)

        # Replace {expression} with %s
        format_str = re.sub(pattern, "%s", f_string)

        return format_str, matches

    def safe_command_for_subprocess(
        self, directory: str, extra_args: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build safe command arguments for subprocess calls using shlex.

        Args:
            directory: Target directory path
            extra_args: Additional arguments

        Returns:
            List of safely escaped command arguments
        """
        # Use the current script name instead of __file__ which may not be available
        script_name = "log_lazy_formatter.py"
        cmd = [script_name, directory]
        if extra_args:
            cmd.extend(extra_args)
        return [shlex.quote(arg) for arg in cmd]

    def should_skip_method_call(self, method_name: str) -> bool:
        """
        Check if a method call should be skipped (not converted).

        Args:
            method_name: The method name being called (e.g., 'info', 'getLogger')

        Returns:
            True if the method should be skipped, False if it should be converted
        """
        return (
            method_name in self.config.skip_methods
            or method_name not in self.config.logging_methods
        )

    def is_logger_assignment(self, line: str) -> bool:
        """
        Check if line contains logger assignment/creation that should be skipped.

        Args:
            line: Source code line to check

        Returns:
            True if this is a logger assignment/creation line
        """
        # Common logger assignment patterns
        assignment_patterns = [
            r"^\s*\w+\s*=\s*logging\.getLogger",  # logger = logging.getLogger(...)
            r"^\s*self\.\w+\s*=\s*logging\.getLogger",  # self.logger = logging.getLogger(...)
            r"^\s*\w+\s*=\s*\w+\.getLogger",  # logger = some_module.getLogger(...)
        ]

        return any(re.search(pattern, line.strip()) for pattern in assignment_patterns)

    def convert_logging_line(self, line: str) -> Optional[str]:
        """
        Convert a single line containing logging calls.

        Returns None if no changes needed, otherwise returns modified line.
        """
        # Skip logger assignment/creation lines
        if self.is_logger_assignment(line):
            return None

        # Enhanced pattern for logger.level(f"...") calls - only actual logging methods
        logging_methods_pattern = "|".join(self.config.logging_methods)
        f_string_pattern = (
            rf"(\b(?:logger|log|_logger|self\.logger|self\.log|logging)\."
            rf'(?:{logging_methods_pattern}))\s*\(\s*f(["\'])([^"\']*)\2([^)]*)\)'
        )

        def replace_f_string(match: re.Match[str]) -> str:
            method_call = match.group(1)
            quote_char = match.group(2)
            f_content = match.group(3)
            remaining_args = match.group(4)

            # Extract the method name to double-check
            method_name = method_call.split(".")[-1]
            if self.should_skip_method_call(method_name):
                return match.group(0)  # Return original, don't convert

            format_str, variables = self.extract_f_string_content(f_content)

            if not variables:
                # No variables, just remove the f prefix
                return f"{method_call}({quote_char}{format_str}{quote_char}{remaining_args})"

            # Build the new call with lazy formatting
            var_args = ", ".join(variables)
            if remaining_args.strip():
                return (
                    f"{method_call}({quote_char}{format_str}{quote_char}, "
                    f"{var_args}{remaining_args})"
                )
            return f"{method_call}({quote_char}{format_str}{quote_char}, {var_args})"

        # Enhanced pattern for .format() calls - only actual logging methods
        format_pattern = (
            rf"(\b(?:logger|log|_logger|self\.logger|self\.log|logging)\."
            rf'(?:{logging_methods_pattern}))\s*\(\s*(["\'])([^"\']*)\2\.format\(([^)]*)\)([^)]*)\)'
        )

        def replace_format_call(match: re.Match[str]) -> str:
            method_call = match.group(1)
            quote_char = match.group(2)
            format_content = match.group(3)
            format_args = match.group(4)
            remaining_args = match.group(5)

            # Extract the method name to double-check
            method_name = method_call.split(".")[-1]
            if self.should_skip_method_call(method_name):
                return match.group(0)  # Return original, don't convert

            # Convert {} to %s and {name} to %(name)s
            if "{}" in format_content:
                lazy_format = format_content.replace("{}", "%s")
                return (
                    f"{method_call}({quote_char}{lazy_format}{quote_char}, "
                    f"{format_args}{remaining_args})"
                )
            # Handle named placeholders
            lazy_format = re.sub(r"\{(\w+)\}", r"%(\1)s", format_content)
            return (
                f"{method_call}({quote_char}{lazy_format}{quote_char}, "
                f"{format_args}{remaining_args})"
            )

        original_line = line

        # Apply f-string conversion
        line = re.sub(f_string_pattern, replace_f_string, line)

        # Apply .format() conversion
        line = re.sub(format_pattern, replace_format_call, line)

        return line if line != original_line else None

    def process_file(self, file_path: Path) -> int:
        """
        Process a single Python file and convert logging calls.

        Returns:
            Number of changes made in this file
        """
        try:
            with file_path.open(encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            module_logger.warning("Could not decode %s, skipping...", file_path)
            return 0
        except OSError as e:
            module_logger.error("Error reading %s: %s", file_path, e)
            return 0

        modified_lines: List[str] = []
        changes_in_file = 0

        for line_num, line in enumerate(lines, 1):
            converted_line = self.convert_logging_line(line)
            if converted_line is not None:
                modified_lines.append(converted_line)
                changes_in_file += 1
                if self.dry_run:
                    module_logger.info("%s:%s", file_path, line_num)
                    module_logger.info("  OLD: %s", line.strip())
                    module_logger.info("  NEW: %s", converted_line.strip())
                    module_logger.info("")
            else:
                modified_lines.append(line)

        if changes_in_file > 0 and not self.dry_run:
            # Create backup if requested
            if self.backup:
                import shutil

                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                try:
                    shutil.copy2(file_path, backup_path)
                    module_logger.info("Backup created: %s", backup_path)
                except Exception as e:
                    module_logger.warning(
                        "Failed to create backup for %s: %s", file_path, e
                    )

            # Write modified content
            with file_path.open("w", encoding="utf-8") as f:
                f.writelines(modified_lines)

            module_logger.info("Modified %s: %s changes", file_path, changes_in_file)

        return changes_in_file

    def _process_files_multithreaded(
        self, python_files: List[Path], threads: Optional[int]
    ) -> None:
        """
        Process Python files using multiple threads.

        Args:
            python_files: List of Python files to process
            threads: Number of threads to use
        """

        # Thread-safe counters
        changes_lock = threading.Lock()
        processed_lock = threading.Lock()

        def process_file_threaded(file_path: Path) -> int:
            """Process a single file and update thread-safe counters."""
            changes = self.process_file(file_path)
            if changes > 0:
                with processed_lock:
                    self.files_processed += 1
            with changes_lock:
                self.changes_made += changes
            return changes

        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file_threaded, file_path): file_path
                for file_path in python_files
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    changes = future.result()
                    if changes > 0:
                        module_logger.info(
                            "  ✓ %s: %d changes",
                            file_path.relative_to(self.root_path)
                            if self.root_path
                            else file_path,
                            changes,
                        )
                except (RuntimeError, ValueError, OSError) as e:
                    module_logger.error("Error processing %s: %s", file_path, e)

                if (i + 1) % 50 == 0:
                    module_logger.info(
                        "Processed %d/%d files...", i + 1, len(python_files)
                    )

    def process(self, directory: str | Path, threads: Optional[int] = None) -> None:
        """
        Convert all Python files in a directory and subdirectories.

        Args:
            directory: Path to the directory to process
            threads: Number of threads to use for processing
        """
        self.root_path = Path(directory).resolve()
        root_path = self.root_path

        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        module_logger.info(
            "%sProcessing directory: %s", "DRY RUN: " if self.dry_run else "", root_path
        )
        module_logger.info(
            "Excluding directories: %s", ", ".join(sorted(self.exclude_dirs))
        )
        module_logger.info("Using %d threads", threads)
        module_logger.info("")

        python_files = self.find_python_files(root_path)
        module_logger.info("Found %s Python files to process", len(python_files))
        module_logger.info("")

        if threads is None or threads <= 1:
            # Single-threaded processing
            for file_path in python_files:
                changes = self.process_file(file_path)
                self.changes_made += changes
                if changes > 0:
                    self.files_processed += 1
                    module_logger.info(
                        "  ✓ %s: %d changes", file_path.relative_to(root_path), changes
                    )
        else:
            # Multi-threaded processing
            self._process_files_multithreaded(python_files, threads)

        module_logger.info("\nSummary:")
        module_logger.info("  Files processed: %s", self.files_processed)
        module_logger.info("  Total changes: %s", self.changes_made)
        if self.dry_run:
            module_logger.info(
                "  (This was a dry run - no files were actually modified)"
            )


def log_lazy(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args: Union[str, int, float],
    **kwargs: Union[str, int, float],
) -> None:
    """
    Ensure logger messages use lazy % formatting.

    Examples:
        log_lazy(logger, logging.INFO, f"Value is {x}")      # → logger.info("%s", f"Value is {x}")
        log_lazy(logger, logging.INFO, "Value is {}", x)     # → logger.info("Value is %s", x)
        log_lazy(logger, logging.INFO, "Value is {val}", val=x)
        # → logger.info("Value is %(val)s", {"val": x})
    """
    # Case 1: Python f-string (already interpolated) → wrap as %s
    if not args and not kwargs and "{" not in msg:
        logger.log(level, "%s", msg)
        return

    # Case 2: "{}" placeholders → convert to %s
    if "{}" in msg and not kwargs:
        fmt = msg.replace("{}", "%s")
        logger.log(level, fmt, *args)
        return

    # Case 3: Named placeholders → convert to %(key)s
    if kwargs:
        fmt = re.sub(r"\{(\w+)\}", r"%(\1)s", msg)
        logger.log(level, fmt, kwargs)
        return

    # Case 4: Mixed or complex formatting → fallback to %s
    logger.log(level, "%s", msg)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert f-string and .format() logging calls to lazy % formatting",
        prog="log_lazy_formatter" if __name__ != "__main__" else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
╔══════════════════════════════════════════════════════════════════════════════╗
                               📋 EXAMPLES
╠══════════════════════════════════════════════════════════════════════════════╣
 Basic conversion with dry run:
   log-lazy-formatter /path/to/project --dry-run

 Convert with custom exclusions:
   log-lazy-formatter /path/to/project --exclude tests --exclude docs

 Convert without backups:
   log-lazy-formatter /path/to/project --no-backup

 Verbose conversion:
   log-lazy-formatter /path/to/project --verbose

 Full conversion with all options:
   log-lazy-formatter /path/to/project --exclude build
     --exclude dist --verbose
╚══════════════════════════════════════════════════════════════════════════════╝

📖 CONFIGURATION GROUPS:
  Directory Processing - Specify target directory and exclusions
  Output Control       - Backup and dry-run options
  Display Options      - Verbose output and progress information

🎯 QUICK START:
  1. Run dry-run first: log-lazy-formatter /path/to/project --dry-run
  2. Review changes and exclude directories if needed: --exclude <dir>
  3. Run actual conversion: log-lazy-formatter /path/to/project
  4. Use verbose mode for detailed progress: --verbose

🔐 SECURITY NOTES:
  • Always run --dry-run first to review changes
  • Backup files are created by default (.bak extension)
  • Use --exclude to avoid processing sensitive directories
  • Review converted logging calls for correctness

Common logger patterns that will be converted:
  logger.info("Processing %s", file)     → logger.info("Processing %s", file)
  log.debug("Status: %s", x)             → log.debug("Status: %s", x)
  self.logger.error("Error: %s", err)    → self.logger.error("Error: %s", err)
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
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress information",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=4,
        metavar="N",
        help="Number of threads to use for processing (default: 4)",
    )
    return parser


def convert_log_format(
    directory: str | Path,
    exclude_dirs: Optional[Set[str]],
    backup: bool,
    dry_run: bool,
    threads: Optional[int] = None,
) -> int:
    """Run the log format conversion on the specified directory."""
    converter = LogFormatConverter(
        exclude_dirs=exclude_dirs, backup=backup, dry_run=dry_run
    )

    try:
        converter.process(directory, threads=threads)
    except (FileNotFoundError, ValueError) as e:
        module_logger.error("Error: %s", e)
        return 1
    except KeyboardInterrupt:
        module_logger.info("\nOperation cancelled by user")
        return 1
    except Exception as e:
        module_logger.error("Unexpected error: %s", e)

        traceback.print_exc()
        return 1

    return 0


def main() -> int:
    """Command line interface for the log format converter."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Use shlex to safely handle directory paths with spaces
    directory_path = (
        shlex.quote(args.directory) if " " in args.directory else args.directory
    )

    # Handle exclude directories safely
    exclude_dirs = set(args.exclude_dirs) if args.exclude_dirs else None

    # Remove quotes from directory path for actual processing
    actual_directory = (
        shlex.split(directory_path)[0]
        if directory_path.startswith('"')
        else directory_path
    )

    return convert_log_format(
        actual_directory, exclude_dirs, not args.no_backup, args.dry_run, args.threads
    )


def example_usage() -> None:
    """Example usage of the log lazy converter."""
    # Clear console
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "cls"], check=False)
    else:
        subprocess.run(["clear"], check=False)

    # Mock sys.argv for demonstration
    original_argv = sys.argv.copy()
    sys.argv = shlex.split("log_lazy_formatter intelag_pkg_manager --dry-run --verbose")

    try:
        # Run main function as if called from command line
        exit_code = main()
        if exit_code != 0:
            module_logger.error("Example usage failed with exit code: %s", exit_code)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
