"""
╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-17 10:32:11
╰────────────────────────────────────────────────────────────────────────────────────────╯

Configuration constants and defaults for the type annotation fixer.
"""

# Standard library imports
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set

# Third party imports
# None

# Internal imports
# Removed intelag_pkg_manager dependency


class TypeAnnotationFixerConfig:
    """
    Configuration class for the type annotation fixer.

    This class manages all configuration options for the type annotation fixer,
    including directories to exclude, file extensions, type mappings, and runtime options.
    """

    # Default directories to exclude
    DEFAULT_EXCLUDE_DIRS: ClassVar[Set[str]] = {
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".tox",
        "build",
        "dist",
        ".egg-info",
        ".mypy_cache",
        "htmlcov",
        "coverage_html",
        ".coverage",
        ".idea",
        ".vscode",
        "migrations",
        ".docker",
        "docker",
        ".intelag",
        "out",
        "bin",
        "obj",
        "MyPy",
        "intelag_packages",
        "intelag_data_collection_manager",
        "intelag_vsix_packages",
        "intelag_vsix_creator",
        "common_media",
        "script",
    }

    # File extensions to process
    PYTHON_EXTENSIONS: ClassVar[Set[str]] = {".py", ".pyw"}

    # Common type mappings for automatic inference
    TYPE_MAPPINGS: ClassVar[Dict[str, str]] = {
        "str": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "list": "List[Any]",
        "dict": "Dict[str, Any]",
        "tuple": "Tuple[Any, ...]",
        "set": "Set[Any]",
        "None": "None",
        "NoneType": "None",
    }

    # Default configuration values
    exclude_dirs: Set[str]
    python_extensions: Set[str]
    type_mappings: Dict[str, str]
    backup: bool
    dry_run: bool
    mypy_config: Optional[str]
    max_file_size: int
    timeout: int
    batch_size: int
    enable_autotyping: bool
    enable_auto_type_annotate: bool
    enable_monkeytype: bool
    enable_pytype: bool

    def __init__(
        self,
        exclude_dirs: Optional[Set[str]] = None,
        python_extensions: Optional[Set[str]] = None,
        type_mappings: Optional[Dict[str, str]] = None,
        backup: bool = True,
        dry_run: bool = False,
        mypy_config: Optional[str] = None,
        max_file_size: int = 100_000,
        timeout: int = 30,
        batch_size: int = 500,
        enable_autotyping: bool = False,
        enable_auto_type_annotate: bool = False,
        enable_monkeytype: bool = False,
        enable_pytype: bool = False,
    ) -> None:
        self.exclude_dirs = exclude_dirs if exclude_dirs is not None else self.DEFAULT_EXCLUDE_DIRS.copy()
        self.python_extensions = python_extensions if python_extensions is not None else self.PYTHON_EXTENSIONS.copy()
        self.type_mappings = type_mappings if type_mappings is not None else self.TYPE_MAPPINGS.copy()
        self.backup = backup
        self.dry_run = dry_run
        self.mypy_config = mypy_config
        self.max_file_size = max_file_size
        self.timeout = timeout
        self.batch_size = batch_size
        self.enable_autotyping = enable_autotyping
        self.enable_auto_type_annotate = enable_auto_type_annotate
        self.enable_monkeytype = enable_monkeytype
        self.enable_pytype = enable_pytype
        self._custom_validation()

    def _custom_validation(self) -> None:
        """Custom validation for type annotation fixer configuration."""
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate that exclude_dirs contains valid directory names
        if not all(d for d in self.exclude_dirs):
            raise ValueError("exclude_dirs must contain only non-empty strings")

        # Validate python_extensions
        if not all(ext.startswith(".") for ext in self.python_extensions):
            raise ValueError("python_extensions must be strings starting with '.'")

    @classmethod
    def create_default(cls) -> "TypeAnnotationFixerConfig":
        """Create a default configuration instance."""
        return cls()

    def get_exclude_dirs_as_list(self) -> List[str]:
        """Get exclude directories as a sorted list."""
        return sorted(self.exclude_dirs)

    def get_python_extensions_as_list(self) -> List[str]:
        """Get Python extensions as a sorted list."""
        return sorted(self.python_extensions)

    def should_exclude_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be excluded based on configuration."""
        return dir_path.name in self.exclude_dirs or dir_path.name.startswith(".")

    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed based on configuration."""
        return file_path.suffix in self.python_extensions

    def is_file_too_large(self, file_path: Path) -> bool:
        """Check if a file is too large to process."""
        try:
            return file_path.stat().st_size > self.max_file_size
        except OSError:
            return False  # If we can't check size, assume it's okay


def example_usage() -> None:
    """Example usage of TypeAnnotationFixerConfig."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Creating default TypeAnnotationFixerConfig...")
    config = TypeAnnotationFixerConfig.create_default()

    logger.info("Exclude directories: %s", config.get_exclude_dirs_as_list())
    logger.info("Python extensions: %s", config.get_python_extensions_as_list())
    logger.info("Backup enabled: %s", config.backup)
    logger.info("Dry run: %s", config.dry_run)


if __name__ == "__main__":
    example_usage()
