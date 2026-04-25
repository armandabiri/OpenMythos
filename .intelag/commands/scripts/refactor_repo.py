"""Apply file, directory, and symbol rename plans to a repository tree."""

import importlib
import logging
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

# Configure logger
logger = logging.getLogger(__name__)


class RenameRule(TypedDict):
    """Single class rename mapping."""

    old: str
    new: str


class ClassRule(TypedDict):
    """Class rename rules for a source file."""

    old_rel_file: str
    renames: list[RenameRule]


def _load_yaml_mapping(file_path: Path) -> dict[str, object]:
    """Load a YAML file and return a mapping-like object."""
    yaml_module = importlib.import_module("yaml")
    safe_load_obj = getattr(yaml_module, "safe_load", None)
    if not callable(safe_load_obj):
        raise ValueError("yaml.safe_load is not available")
    with file_path.open(encoding="utf-8") as yaml_file:
        loaded = safe_load_obj(yaml_file)
    if isinstance(loaded, dict):
        loaded_dict = cast(dict[object, object], loaded)
        return {str(key): value for key, value in loaded_dict.items()}
    return {}


def remove_readonly(func: Callable[[str], None], path: str, _: object) -> None:
    """Error handler for shutil.rmtree to handle read-only files."""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def refactor(
    repo_root: str | Path,
    plan_file: str | Path,
    dry_run: bool = True,
    output_suffix: str = "",
) -> None:
    """Execute refactor operations described in a YAML plan."""
    logger.info("Starting refactor process...")
    logger.info("  Repo Root: %s", repo_root)
    logger.info("  Plan File: %s", plan_file)
    logger.info("  Dry Run: %s", dry_run)
    logger.info("  Output Suffix: '%s'", output_suffix)
    refactor_plan = _load_yaml_mapping(Path(plan_file))
    logger.info("Plan loaded successfully.")

    repo_root = Path(repo_root).resolve()
    target_scope = refactor_plan.get("target_scope")

    # 1. Normalize renames from the plan
    def normalize_list_or_dict(
        data: object,
        old_key: str = "old_rel",
        new_key: str = "new_name",
    ) -> list[dict[str, str]]:
        if not data:
            return []
        if isinstance(data, dict):
            data_dict = cast(dict[object, object], data)
            # Format: { "new_name": "old_rel" }
            return [{old_key: str(value), new_key: str(key)} for key, value in data_dict.items()]
        if isinstance(data, list):
            data_list = cast(list[object], data)
            res: list[dict[str, str]] = []
            for item in data_list:
                if isinstance(item, str) and "=>" in item:
                    # Format: "old_rel => new_name"
                    parts = item.split("=>", 1)
                    res.append({old_key: parts[0].strip(), new_key: parts[1].strip()})
                elif isinstance(item, dict):
                    item_dict = cast(dict[object, object], item)
                    # Format: { "new_name": "old_rel" }
                    for key, value in item_dict.items():
                        res.append({old_key: str(value), new_key: str(key)})
            return res
        return []

    directories_norm = normalize_list_or_dict(refactor_plan.get("directories"))
    files_norm = normalize_list_or_dict(refactor_plan.get("files"))
    logger.info(
        "Normalized Rules: %d directory renames, %d file renames.",
        len(directories_norm),
        len(files_norm),
    )

    def normalize_classes(classes_raw: object) -> list[ClassRule]:
        if not classes_raw:
            return []
        res: list[ClassRule] = []
        if not isinstance(classes_raw, list):
            return res
        classes_list = cast(list[object], classes_raw)
        for entry in classes_list:
            if not isinstance(entry, dict):
                continue
            entry_dict = cast(dict[object, object], entry)
            old_rel_file_obj = entry_dict.get("old_rel_file")
            renames_raw = entry_dict.get("renames")
            norm_renames: list[RenameRule] = []
            if isinstance(renames_raw, dict):
                renames_dict = cast(dict[object, object], renames_raw)
                norm_renames = [{"old": str(value), "new": str(key)} for key, value in renames_dict.items()]
            elif isinstance(renames_raw, list):
                renames_list = cast(list[object], renames_raw)
                for r in renames_list:
                    if isinstance(r, str) and "=>" in r:
                        parts = r.split("=>", 1)
                        norm_renames.append({"old": parts[0].strip(), "new": parts[1].strip()})
                    elif isinstance(r, dict):
                        rename_dict = cast(dict[object, object], r)
                        for key, value in rename_dict.items():
                            norm_renames.append({"old": str(value), "new": str(key)})
            if isinstance(old_rel_file_obj, str) and norm_renames:
                res.append({"old_rel_file": old_rel_file_obj, "renames": norm_renames})
        return res

    classes_norm = normalize_classes(refactor_plan.get("classes"))
    logger.info("Normalized Class Rules: %d entries.", len(classes_norm))

    def get_target_path(old_rel_path: str) -> str:
        parts = old_rel_path.split("/")
        new_parts: list[str] = []
        current_old_prefix = ""

        for i, part in enumerate(parts):
            if current_old_prefix:
                current_old_prefix += "/" + part
            else:
                current_old_prefix = part

            # Check if this prefix has a rename in directories
            renamed = False
            for d in directories_norm:
                if d["old_rel"] == current_old_prefix:
                    new_parts.append(d["new_name"])
                    renamed = True
                    break
            if not renamed:
                # Check if it's the last part and it's a file
                if i == len(parts) - 1:
                    file_renamed = False
                    for file_entry in files_norm:
                        if file_entry["old_rel"] == old_rel_path:
                            new_parts.append(file_entry["new_name"])
                            file_renamed = True
                            break
                    if not file_renamed:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
        return "/".join(new_parts)

    full_repo_mapping: dict[str, str] = {}  # old_rel -> new_rel
    # Walk the entire repo to build a complete module map
    # Get exclusions from environment or use defaults
    env_exclude_folders = os.environ.get("INTELAG_DASHBOARD_EXCLUDE_FOLDERS", "")
    if env_exclude_folders:
        exclude_folders = [f.strip() for f in env_exclude_folders.split(",") if f.strip()]
    else:
        exclude_folders = [
            ".git",
            ".venv",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "node_modules",
        ]

    env_exclude_files = os.environ.get("INTELAG_DASHBOARD_EXCLUDE_FILES", "")
    exclude_files = [f.strip() for f in env_exclude_files.split(",") if f.strip()]

    # Walk the entire repo to build a complete module map
    for root, _dirs, files in os.walk(repo_root):
        # Skip excluded folders
        # Check if any part of the path matches an excluded folder
        rel_root = Path(root).relative_to(repo_root).as_posix()
        path_parts = rel_root.split("/")

        if any(ex in path_parts for ex in exclude_folders):
            continue

        # Also check hardcoded ignores if not covered above (imports are relative to root)
        # The previous 'ignored in root' was a substring check which is risky (e.g. 'git' in 'digital')
        # We now use path component matching which is safer

        for f in files:
            if f in exclude_files:
                continue

            full_path = Path(root) / f
            rel_path = full_path.relative_to(repo_root).as_posix()
            full_repo_mapping[rel_path] = get_target_path(rel_path)

    logger.info("Scanned repo. Found %d files.", len(full_repo_mapping))

    # 2. Build Module Mapping (for imports) from full_repo_mapping
    module_map: dict[str, str] = {}
    for old_rel, new_rel in full_repo_mapping.items():
        if old_rel.endswith(".py"):
            old_mod = old_rel[:-3].replace("/", ".")
            new_mod = new_rel[:-3].replace("/", ".")
            if old_mod != new_mod:
                module_map[old_mod] = new_mod
        elif old_rel.endswith("__init__.py"):
            # Handle package renames
            old_mod = old_rel.replace("/__init__.py", "").replace("/", ".")
            new_mod = new_rel.replace("/__init__.py", "").replace("/", ".")
            if old_mod != new_mod:
                module_map[old_mod] = new_mod

    logger.info("Generated %d module import replacements.", len(module_map))

    # 3. Determine files to process based on target_scope
    files_to_process: dict[str, str] = {}
    if isinstance(target_scope, str) and target_scope:
        files_to_process = {rel: new for rel, new in full_repo_mapping.items() if rel.startswith(target_scope)}
    else:
        files_to_process = full_repo_mapping

    logger.info("Processing %d files (Scope: %s).", len(files_to_process), target_scope or "All")

    # 3. Build Class Mapping
    class_map: dict[str, str] = {}  # old_class -> new_class
    for entry in classes_norm:
        for class_rename in entry["renames"]:
            class_map[class_rename["old"]] = class_rename["new"]

    logger.info("Generated %d class/symbol replacements.", len(class_map))

    # 4. Perform Search and Replace in all files
    # We should search for:
    # - exact class names (with word boundaries)
    # - module paths in imports

    def refactor_content(content: str) -> str:
        # Replace class names
        for old_c, new_c in class_map.items():
            content = re.sub(r"\b" + re.escape(old_c) + r"\b", new_c, content)

        # Replace module paths (imports)
        # We should sort module_map by length descending to avoid partial matches
        sorted_modules = sorted(module_map.keys(), key=len, reverse=True)
        for old_m in sorted_modules:
            new_m = module_map[old_m]
            # Replace in 'import a.b.c' or 'from a.b.c import ...'
            content = re.sub(r"\b" + re.escape(old_m) + r"\b", new_m, content)

        return content

    # If dry_run, we copy to a new folder
    if dry_run:
        target_root = repo_root.parent / (repo_root.name + output_suffix)
        if target_root.exists():
            shutil.rmtree(target_root, onerror=remove_readonly)
        target_root.mkdir(parents=True)
    else:
        target_root = repo_root

    # Process selected files
    files_modified_count = 0
    for old_rel, new_rel in files_to_process.items():
        # Extra safety: skip hidden folders in processing
        if any(part.startswith(".") and part != "." for part in old_rel.split("/")):
            continue

        old_path = repo_root / old_rel
        new_path = target_root / new_rel

        new_path.parent.mkdir(parents=True, exist_ok=True)

        if old_path.suffix in [".py", ".yaml", ".txt", ".md"]:
            with Path(old_path).open(encoding="utf-8", errors="ignore") as content_f:
                content = content_f.read()

            new_content = refactor_content(content)

            if content != new_content:
                logger.info("MODIFIED: %s", new_rel)
                files_modified_count += 1

            with Path(new_path).open("w", encoding="utf-8") as output_f:
                output_f.write(new_content)
        else:
            # Binary or other files
            # Check if we are copying to the same file
            if old_path.resolve() != new_path.resolve():
                shutil.copy2(old_path, new_path)

    logger.info("Refactoring complete. Results in %s", target_root)
    logger.info("Total files modified: %d", files_modified_count)


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: refactor_repo.py <plan_yaml> [repo_dir] [--apply]")


def main() -> int:
    """Run CLI argument parsing and invoke refactor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--example" in sys.argv:
        example_usage()
        return 0

    if len(sys.argv) < 2:
        logger.error("Usage: python refactor_repo.py <plan_yaml> [repo_dir] [--apply]")
        return 1

    arg1 = Path(sys.argv[1])
    repo_path = None
    plan_path = None

    # Heuristic to detect argument order
    if arg1.is_file() and arg1.suffix in [".yaml", ".yml"]:
        # Format: plan_yaml [repo_dir]
        plan_path = arg1
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
            repo_path = Path(sys.argv[2])
    else:
        # Assume Old Format: repo_dir plan_yaml
        repo_path = arg1
        if len(sys.argv) > 2:
            plan_path = Path(sys.argv[2])

    if not plan_path or not plan_path.exists():
        logger.error("Plan file not found: %s", plan_path)
        return 1

    # Read config from YAML
    config = _load_yaml_mapping(plan_path)

    # Resolve Repo Path
    if not repo_path:
        config_repo_root = config.get("repo_root")
        if isinstance(config_repo_root, str) and config_repo_root:
            repo_path = Path(config_repo_root)
        else:
            repo_path = Path.cwd()

    # Resolve Dry Run
    # CLI flag overrides YAML
    apply_changes_cli = "--apply" in sys.argv
    apply_changes_yaml = config.get("apply_changes", False)

    # If CLI flag is present, it forces apply.
    # Otherwise, trust YAML. If neither, default to dry run.
    should_apply = apply_changes_cli or apply_changes_yaml

    output_suffix_val = "_validation" if not should_apply else ""

    refactor(repo_path, plan_path, dry_run=not should_apply, output_suffix=output_suffix_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
