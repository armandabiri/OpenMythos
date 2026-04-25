"""
Apply STANDARD INTELAG RULES from .intelag/python/pyproject.toml into a target TOML file.

Replaces existing tool.ruff, tool.mypy, tool.pytest.ini_options sections in the target
with the standard rules; leaves all other content unchanged.

Usage:
  python apply_intelag_python_rules.py --target path/to/pyproject.toml [--workspace-root ROOT]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

STANDARD_MARKER = "STANDARD INTELAG RULES"
TEMPLATE_REL_PATH = ".intelag/python/pyproject.toml"

# Section headers that belong to standard rules (we remove these from target then insert template block).
STANDARD_SECTION_HEADERS = [
    r"^\[tool\.ruff\]",
    r"^\[tool\.ruff\.lint\]",
    r"^\[tool\.ruff\.lint\.isort\]",
    r"^\[tool\.ruff\.format\]",
    r"^\[tool\.mypy\]",
    r"^\[\[tool\.mypy\.overrides\]\]",
    r"^\[tool\.ruff\.lint\.per-file-ignores\]",
    r"^\[tool\.pytest\.ini_options\]",
]
SECTION_HEADER_RE = re.compile(r"^(\[\[?[a-zA-Z0-9_.]+\]\]?)\s*$")


def find_workspace_root(start: Path) -> Path | None:
    """Walk up from start to find a directory containing .intelag/python/pyproject.toml."""
    current = start.resolve()
    for _ in range(20):
        if (current / TEMPLATE_REL_PATH).exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def extract_standard_block(template_path: Path) -> str:
    """Extract from template the block from STANDARD INTELAG RULES to end of file."""
    text = template_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if STANDARD_MARKER in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Template {template_path} has no '{STANDARD_MARKER}' marker")
    # Find first [tool.*] or [[tool.*]] after the marker
    for i in range(start_idx + 1, len(lines)):
        if re.match(r"^\[\[?tool\.", lines[i]):
            return "".join(lines[i:])
    raise ValueError(f"Template {template_path}: no [tool.*] section after '{STANDARD_MARKER}'")


def is_standard_section_header(line: str) -> bool:
    """True if this line is one of the standard section headers we replace."""
    line_stripped = line.strip()
    for pat in STANDARD_SECTION_HEADERS:
        if re.match(pat, line_stripped):
            return True
    return False


def remove_standard_sections(content: str) -> str:
    """Remove from content any block whose header is in STANDARD_SECTION_HEADERS. Returns new content."""
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    skip_until_next_section = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if SECTION_HEADER_RE.match(line.strip()):
            if is_standard_section_header(line):
                skip_until_next_section = True
                i += 1
                continue
            skip_until_next_section = False
        if not skip_until_next_section:
            out.append(line)
        i += 1
    result = "".join(out)
    # Normalize: single trailing newline, no trailing blank lines from removed blocks
    return result.rstrip() + "\n"


def apply_rules(target_path: Path, standard_block: str) -> None:
    """Remove standard sections from target file and append standard_block."""
    content = target_path.read_text(encoding="utf-8")
    cleaned = remove_standard_sections(content)
    # Ensure one newline before appended block
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    if not standard_block.startswith("\n"):
        standard_block = "\n" + standard_block
    new_content = cleaned.rstrip() + "\n\n# Applied from Intelag standard rules\n" + standard_block
    target_path.write_text(new_content, encoding="utf-8")
    logger.info("Applied standard Intelag rules to %s", target_path)


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Apply STANDARD INTELAG RULES from .intelag/python/pyproject.toml into a TOML file.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to the target pyproject.toml (or other TOML) to update.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Workspace root (default: auto-detect from target path).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    target = args.target.resolve()
    if not target.exists():
        logger.error("Target file not found: %s", target)
        return 1

    root = args.workspace_root
    if root is None:
        root = find_workspace_root(target.parent)
    else:
        root = root.resolve()
    if not root:
        logger.error(
            "Workspace root not found (no %s above %s). Pass --workspace-root.",
            TEMPLATE_REL_PATH,
            target,
        )
        return 1

    template_path = root / TEMPLATE_REL_PATH
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        return 1

    try:
        standard_block = extract_standard_block(template_path)
        apply_rules(target, standard_block)
        return 0
    except (ValueError, OSError) as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
