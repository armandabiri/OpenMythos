"""
╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-17 10:32:11
╰────────────────────────────────────────────────────────────────────────────────────────╯

Core functionality for the type annotation fixer.
"""

# Standard library imports
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from modules.config import TypeAnnotationFixerConfig
    from modules.fixer_fixing import TypeFixingMixin
    from modules.fixer_inference import TypeInferenceMixin
except ImportError:
    from .modules.config import TypeAnnotationFixerConfig
    from .modules.fixer_fixing import TypeFixingMixin
    from .modules.fixer_inference import TypeInferenceMixin

logger = logging.getLogger(__name__)


class TypeAnnotationFixer(TypeInferenceMixin, TypeFixingMixin):
    """
    Automatically fix missing type annotations using mypy analysis.
    """

    def __init__(
        self,
        config: Optional[TypeAnnotationFixerConfig] = None,
        exclude_dirs: Optional[Set[str]] = None,
        backup: Optional[bool] = None,
        dry_run: Optional[bool] = None,
        mypy_config: Optional[str] = None,
    ) -> None:
        """
        Initialize the type annotation fixer.

        Args:
            config: Configuration object. If provided, other parameters are ignored.
            exclude_dirs: Additional directories to exclude
            backup: Whether to create backups before modification
            dry_run: If True, only show what would be changed
            mypy_config: Path to mypy configuration file
        """
        if config is None:
            config = TypeAnnotationFixerConfig()
            if exclude_dirs is not None:
                config.exclude_dirs = config.exclude_dirs.union(exclude_dirs)
            if backup is not None:
                config.backup = backup
            if dry_run is not None:
                config.dry_run = dry_run
            if mypy_config is not None:
                config.mypy_config = mypy_config

        self.config = config
        self.exclude_dirs = config.exclude_dirs
        self.backup = config.backup
        self.dry_run = config.dry_run
        self.mypy_config = config.mypy_config
        self.changes_made: int = 0
        self.files_processed: int = 0
        self.mypy_executable: str = self._find_mypy()
        self.root_path: Optional[Path] = None

    def _find_mypy(self) -> str:
        """Find mypy executable in the system."""
        for cmd in ["mypy", "python -m mypy"]:
            try:
                result = subprocess.run(
                    [*cmd.split(), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=300.0,
                    check=False,
                )  # nosec
                if result.returncode == 0:
                    return cmd
            except (
                subprocess.TimeoutExpired,
                subprocess.CalledProcessError,
                FileNotFoundError,
            ):
                continue
        return "python -m mypy"

    def find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory, respecting exclusions."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Resolve relative dirs to handle exclusions correctly
            rel_root = Path(root).relative_to(directory)

            # Filter out excluded directories
            dirs[:] = [
                d
                for d in dirs
                if d not in self.exclude_dirs
                and not any((rel_root / d).match(exclude) for exclude in self.exclude_dirs)
            ]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def run_mypy_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run mypy analysis on a single file and return issues."""
        cmd = [
            *self.mypy_executable.split(),
            str(file_path),
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
        ]

        if self.mypy_config:
            cmd.extend(["--config-file", self.mypy_config])

        try:
            # Use a longer timeout for mypy as it can be slow
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300.0, check=False)  # nosec
            return self.parse_mypy_output(result.stdout)
        except subprocess.TimeoutExpired:
            logger.error("Mypy analysis timed out for %s", file_path)
            return []
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error running mypy: %s", e)
            return []

    def parse_mypy_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse mypy output into a list of issues."""
        issues = []
        # Example format: file.py:10: error: Function is missing a return type annotation [no-untyped-def]
        pattern = re.compile(r"^([^:]+):(\d+): (error|warning|note): (.*) \[([^\]]+)\]$")

        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                issues.append(
                    {
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "type": match.group(3),
                        "message": match.group(4),
                        "code": match.group(5),
                    }
                )

        return issues

    def process_file_batch(self, file_paths: List[Path]) -> int:
        """Process a batch of files and fix type annotations."""
        if not file_paths:
            return 0

        # Run mypy on all files in the batch for efficiency
        cmd = [
            *self.mypy_executable.split(),
            *[str(f) for f in file_paths],
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
        ]

        if self.mypy_config:
            cmd.extend(["--config-file", self.mypy_config])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600.0, check=False)  # nosec
            batch_issues = self.parse_mypy_output(result.stdout)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error running mypy batch: %s", e)
            return 0

        # Group issues by file
        file_issues: Dict[Path, List[Dict[str, Any]]] = {f: [] for f in file_paths}
        for issue in batch_issues:
            # Handle potential relative paths in mypy output
            issue_file = Path(issue["file"]).resolve()
            for f in file_paths:
                if f.resolve() == issue_file:
                    file_issues[f].append(issue)
                    break

        total_fixed = 0
        for file_path, issues in file_issues.items():
            if issues:
                fixed = self.fix_missing_annotations(file_path, issues)
                if fixed > 0:
                    total_fixed += fixed
                    self.changes_made += fixed
            self.files_processed += 1

        return total_fixed

    def process_file(self, file_path: Path) -> int:
        """Process a single file and fix type annotations."""
        issues = self.run_mypy_analysis(file_path)
        if not issues:
            return 0

        return self.fix_missing_annotations(file_path, issues)

    def _run_external_tools(self, root_path: Path) -> None:
        """Run external type annotation tools based on configuration."""
        if self.config.enable_autotyping:
            logger.info("Running autotyping...")
            cmd = ["python", "-m", "autotyping", "--none-return", "--safe", str(root_path)]
            self._execute_tool_cmd(cmd, "autotyping")

        if self.config.enable_auto_type_annotate:
            logger.info("Running auto-type-annotate...")
            # This tool usually requires dmypy
            try:
                subprocess.run(["dmypy", "run"], check=False)  # nosec
                cmd = ["auto-type-annotate", "--application-directories", ".:src", str(root_path)]
                self._execute_tool_cmd(cmd, "auto-type-annotate")
            except FileNotFoundError:
                logger.warning("dmypy not found, skipping auto-type-annotate")

        if self.config.enable_monkeytype:
            # Check for bundled monkeytype_apply.py script
            monkeytype_script = root_path / ".intelag" / "commands" / "scripts" / "monkeytype_apply.py"
            if monkeytype_script.exists():
                self._execute_tool_cmd(
                    [sys.executable, str(monkeytype_script), "--python-exe", sys.executable, "--path", str(root_path)],
                    "monkeytype_apply_script",
                )
            else:
                self._execute_tool_cmd([sys.executable, "-m", "monkeytype", "apply", "."], "monkeytype")
            logger.info("  Note: MonkeyType works best when running specific tests. Using basic check.")

        if self.config.enable_pytype:
            logger.info("Running pytype...")
            cmd = ["pytype", "--protocols", str(root_path)]
            self._execute_tool_cmd(cmd, "pytype")

    def _execute_tool_cmd(self, cmd: List[str], tool_name: str) -> None:
        """Execute a tool command and log results."""
        if self.dry_run:
            logger.info("  [DRY RUN] Would execute: %s", " ".join(cmd))
            return

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )  # nosec
            if result.returncode == 0:
                logger.info("  ✓ %s completed successfully", tool_name)
            else:
                logger.warning("  ⚠ %s returned exit code %d", tool_name, result.returncode)
                if result.stderr:
                    logger.debug("  Error output: %s", result.stderr.strip())
        except FileNotFoundError:
            logger.error("  ❌ %s not found. Please install it.", tool_name)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("  ❌ Error running %s: %s", tool_name, e)

    def process_directory(self, directory: Union[str, Path]) -> None:
        """
        Process all Python files in a directory and fix type annotations.
        """
        self.root_path = Path(directory).resolve()
        root_path = self.root_path

        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Try to run mypy --version to check if it's installed
        try:
            subprocess.run(
                [self.mypy_executable, "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )  # nosec
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.error("Mypy not found or not working. Please install it: pip install mypy")
            return

        logger.info(
            "%sFixing type annotations in: %s",
            "DRY RUN: " if self.dry_run else "",
            root_path,
        )
        logger.info("Using mypy: %s", self.mypy_executable)
        logger.info("Excluding: %s", ", ".join(sorted(self.exclude_dirs)))
        logger.info("")

        python_files = self.find_python_files(root_path)
        if not python_files:
            logger.info("No Python files found to process.")
            return

        logger.info("Found %d Python files to analyze.", len(python_files))

        # Run external tools first if enabled
        self._run_external_tools(root_path)

        # Split into batches (Windows has ~8191 char command-line limit; avoid WinError 206)
        batch_size = getattr(self.config, "batch_size", 100)
        if sys.platform == "win32":
            batch_size = min(batch_size, 50)
        file_batches = [python_files[i : i + batch_size] for i in range(0, len(python_files), batch_size)]

        for i, batch in enumerate(file_batches):
            logger.info(
                "Processing batch %d/%d (%d files)...",
                i + 1,
                len(file_batches),
                len(batch),
            )
            self.process_file_batch(batch)

        logger.info("")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("Type Annotation Fixer Summary:")
        logger.info("  Files processed: %d", self.files_processed)
        logger.info("  Total changes made: %d", self.changes_made)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    """Main entry point for the tool."""
    from modules.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
