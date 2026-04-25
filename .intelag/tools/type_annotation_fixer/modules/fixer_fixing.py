"""
╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-17 10:32:11
╰────────────────────────────────────────────────────────────────────────────────────────╯

Type annotation fixing functionality for the type annotation fixer.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

logger = logging.getLogger(__name__)


class TypeFixingMixin:
    """
    Mixin class for type annotation fixing functionality.
    """

    def fix_missing_annotations(
        self,
        file_path: Path,
        issues: List[Dict[str, Any]],
        dry_run: bool = False,
        backup: bool = True,
    ) -> int:
        """
        Fix missing type annotations in a file based on mypy issues.
        """
        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception as e:
            logger.error("Error reading %s: %s", file_path, e)
            return 0

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error("Syntax error in %s, skipping: %s", file_path, e)
            return 0

        changes = 0
        added_annotations: List[str] = []

        annotation_issues = [
            issue
            for issue in issues
            if "missing" in issue.get("message", "").lower()
            and "annotation" in issue.get("message", "").lower()
        ]

        if not annotation_issues:
            return 0

        current_lines = list(lines)
        annotation_issues.sort(key=lambda x: x.get("line", 0), reverse=True)

        for issue in annotation_issues:
            line_num = issue.get("line", 0)
            if line_num <= 0 or line_num > len(current_lines):
                continue

            target_node = self._find_node_at_line(tree, line_num)
            if not target_node:
                continue

            original_line = current_lines[line_num - 1]
            fixed_line = original_line

            message = issue.get("message", "").lower()
            if "return type" in message and isinstance(
                target_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                fixed_line = self._add_return_annotation(
                    current_lines, target_node, issue
                )
                if fixed_line != current_lines[target_node.lineno - 1]:
                    changes += 1
            elif "parameter" in message and isinstance(
                target_node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                param_name = self._extract_param_name(issue.get("message", ""))
                if param_name:
                    fixed_line = self._add_param_annotation(
                        current_lines, target_node, param_name
                    )
                    if fixed_line != original_line:
                        current_lines[line_num - 1] = fixed_line
                        changes += 1

            if fixed_line != original_line:
                new_types = self._extract_type_annotations(original_line, fixed_line)
                added_annotations.extend(new_types)

        if changes > 0:
            current_lines = self._ensure_typing_imports(
                current_lines, added_annotations
            )
            new_content = "\n".join(current_lines) + "\n"

            if not dry_run:
                if backup:
                    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                    backup_path.write_text(content, encoding="utf-8")

                with file_path.open("w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info("Fixed %s: %d annotations added", file_path, changes)
            else:
                logger.info("DRY RUN: Would fix %s with %d changes", file_path, changes)

        return changes

    def _find_node_at_line(self, tree: ast.AST, lineno: int) -> Optional[ast.AST]:
        """Find the innermost AST node that starts at the given line."""
        best_node = None
        for node in ast.walk(tree):
            if hasattr(node, "lineno") and cast(Any, node).lineno == lineno:
                best_node = node
        return best_node

    def _extract_param_name(self, message: str) -> Optional[str]:
        match = re.search(r'parameter\s+"([^"]+)"', message)
        return match.group(1) if match else None

    def _add_return_annotation(
        self,
        lines: List[str],
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        _issue: Dict[str, Any],
    ) -> str:
        """Adds ' -> Type' to the function header, handling multi-line."""
        start_idx = node.lineno - 1
        end_idx = node.body[0].lineno - 1

        for i in range(end_idx - 1, start_idx - 1, -1):
            line = lines[i]
            if ":" in line:
                stripped = line.rstrip()
                if stripped.endswith(":"):
                    ret_type = self._guess_return_type_from_name(node.name)
                    lines[i] = line[: line.rfind(":")] + f" -> {ret_type}:"
                    return lines[i]
        return lines[node.lineno - 1]

    def _add_param_annotation(
        self,
        lines: List[str],
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        param_name: str,
    ) -> str:
        """Adds ': Type' to a parameter in the signature."""
        if param_name in ("self", "cls"):
            return lines[node.lineno - 1]

        start_idx = node.lineno - 1
        end_idx = node.body[0].lineno - 1

        param_type = self._guess_param_type_from_name(param_name)
        pattern = rf"\b{re.escape(param_name)}\b(?!\s*:)"

        for i in range(start_idx, end_idx):
            if re.search(pattern, lines[i]):
                lines[i] = re.sub(pattern, f"{param_name}: {param_type}", lines[i])
                return lines[i]
        return lines[node.lineno - 1]

    def _extract_type_annotations(
        self, original_line: str, fixed_line: str
    ) -> List[str]:
        """
        Extract newly added type annotations from the difference between original and fixed lines.
        """
        annotation_patterns = [
            r":\s*([^=\s,)]+)",  # Parameter/variable annotations like ': int'
            r"->\s*([^:]+):",  # Return type annotations like '-> str:'
        ]

        annotations: List[str] = []
        for pattern in annotation_patterns:
            original_matches = set(re.findall(pattern, original_line))
            fixed_matches = set(re.findall(pattern, fixed_line))
            new_annotations = fixed_matches - original_matches
            annotations.extend(new_annotations)

        cleaned_annotations: List[str] = []
        for annotation in annotations:
            cleaned = annotation.strip()
            if cleaned:
                cleaned_annotations.append(cleaned)

        return cleaned_annotations

    def _ensure_typing_imports(
        self, lines: List[str], added_annotations: List[str]
    ) -> List[str]:
        """
        Ensure all required typing imports are present based on added annotations.
        """
        if not added_annotations:
            return lines

        TYPING_IMPORTS = {
            "Any": "Any",
            "Dict": "Dict",
            "List": "List",
            "Set": "Set",
            "Tuple": "Tuple",
            "Union": "Union",
            "Optional": "Optional",
            "Callable": "Callable",
            "Iterator": "Iterator",
            "Iterable": "Iterable",
            "Mapping": "Mapping",
            "Sequence": "Sequence",
            "Type": "Type",
            "TypeVar": "TypeVar",
            "Generic": "Generic",
            "Final": "Final",
            "Literal": "Literal",
            "ClassVar": "ClassVar",
            "NoReturn": "NoReturn",
        }

        needed_imports: Set[str] = set()
        for annotation in added_annotations:
            type_parts = self._parse_type_annotation(annotation)
            for part in type_parts:
                if part in TYPING_IMPORTS:
                    needed_imports.add(TYPING_IMPORTS[part])

        if not needed_imports:
            return lines

        existing_imports = self._find_existing_typing_imports(lines)
        missing_imports = needed_imports - existing_imports

        if not missing_imports:
            return lines

        return self._add_typing_imports(lines, missing_imports)

    def _parse_type_annotation(self, annotation: str) -> Set[str]:
        """Parse a type annotation string."""
        clean_annotation = re.sub(r"\s+", "", annotation)
        typing_parts = re.findall(r"\b[A-Z][a-zA-Z]*", clean_annotation)
        builtin_types = {"str", "int", "float", "bool", "None", "Path"}
        return {part for part in typing_parts if part not in builtin_types}

    def _find_existing_typing_imports(self, lines: List[str]) -> Set[str]:
        """Find existing typing imports in the file."""
        existing_imports: Set[str] = set()
        for line in lines:
            line = line.strip()
            if line.startswith("from typing import"):
                import_part = line.replace("from typing import", "").strip()
                import_part = re.sub(r"[(),]", " ", import_part)
                imports = [imp.strip() for imp in import_part.split() if imp.strip()]
                existing_imports.update(imports)
            elif line.startswith("import typing"):
                return {
                    "Any",
                    "Dict",
                    "List",
                    "Set",
                    "Tuple",
                    "Union",
                    "Optional",
                    "Callable",
                    "Iterator",
                    "Iterable",
                    "Mapping",
                    "Sequence",
                    "Type",
                    "TypeVar",
                    "Generic",
                    "Final",
                    "Literal",
                    "ClassVar",
                    "NoReturn",
                }
        return existing_imports

    def _add_typing_imports(
        self, lines: List[str], missing_imports: Set[str]
    ) -> List[str]:
        """Add missing typing imports to the file."""
        if not missing_imports:
            return lines

        insert_position = self._find_import_insertion_position(lines)
        sorted_imports = sorted(missing_imports)

        updated_lines = lines.copy()
        existing_typing_line = self._find_existing_typing_import_line(lines)

        if existing_typing_line is not None:
            updated_lines[existing_typing_line] = self._extend_typing_import(
                lines[existing_typing_line], missing_imports
            )
        else:
            if len(sorted_imports) == 1:
                import_line = f"from typing import {sorted_imports[0]}"
            else:
                import_line = "from typing import {}".format(", ".join(sorted_imports))
            updated_lines.insert(insert_position, import_line)

        return updated_lines

    def _find_import_insertion_position(self, lines: List[str]) -> int:
        """Find the best position to insert new import statements."""
        last_import_line = -1
        docstring_end = -1
        in_docstring = False
        docstring_quote = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 or (i == 1 and lines[0].strip().startswith("#!")):
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_quote = stripped[:3]
                    if stripped.count(docstring_quote) >= 2:
                        docstring_end = i
                    else:
                        in_docstring = True
                elif stripped.startswith('"') or stripped.startswith("'"):
                    quote = stripped[0]
                    if stripped.count(quote) >= 2:
                        docstring_end = i
            elif (
                in_docstring
                and docstring_quote is not None
                and docstring_quote in stripped
            ):
                in_docstring = False
                docstring_end = i

            if (
                stripped.startswith("import ") or stripped.startswith("from ")
            ) and not in_docstring:
                last_import_line = i
            elif stripped and not stripped.startswith("#") and not in_docstring:
                break

        if last_import_line >= 0:
            return last_import_line + 1
        elif docstring_end >= 0:
            return docstring_end + 1
        else:
            return 0

    def _find_existing_typing_import_line(self, lines: List[str]) -> Optional[int]:
        """Find the line number of existing 'from typing import' statement."""
        for i, line in enumerate(lines):
            if line.strip().startswith("from typing import"):
                return i
        return None

    def _extend_typing_import(self, existing_line: str, new_imports: Set[str]) -> str:
        """Extend an existing typing import line with new imports."""
        import_part = existing_line.replace("from typing import", "").strip()
        has_parens = import_part.startswith("(") and import_part.endswith(")")
        if has_parens:
            import_part = import_part[1:-1].strip()

        current_imports = [imp.strip() for imp in import_part.split(",") if imp.strip()]
        all_imports = set(current_imports) | new_imports
        sorted_imports = sorted(all_imports)
        import_str = ", ".join(sorted_imports)

        if len(sorted_imports) > 3 or len(import_str) > 60:
            return f"from typing import ({import_str})"
        else:
            return f"from typing import {import_str}"

    def _guess_return_type_from_name(self, func_name: str) -> str:
        """Guess return type from function name patterns."""
        name_lower = func_name.lower()
        if name_lower in ("__init__", "__enter__", "__exit__"):
            return "None"
        elif name_lower.startswith(("is_", "has_", "can_", "should_")):
            return "bool"
        elif name_lower.startswith("get_") and "list" in name_lower:
            return "List[Any]"
        elif name_lower.startswith("get_") and "dict" in name_lower:
            return "Dict[str, Any]"
        elif name_lower.endswith("_count") or name_lower.endswith("_size"):
            return "int"
        elif name_lower.endswith("_name") or name_lower.endswith("_str"):
            return "str"
        else:
            return "Any"

    def _guess_param_type_from_name(self, param_name: str) -> str:
        """Guess parameter type from name patterns."""
        name_lower = param_name.lower()
        if name_lower.endswith(("_id", "id", "_count", "_size")):
            return "int"
        elif name_lower.endswith(("_name", "name", "title", "text", "message", "_str")):
            return "str"
        elif name_lower.endswith(("_flag", "_enabled")) or name_lower.startswith("is_"):
            return "bool"
        elif name_lower.endswith(("_list", "_items")) or name_lower in (
            "items",
            "values",
        ):
            return "List[Any]"
        elif name_lower.endswith(("_dict", "_data")) or name_lower in (
            "data",
            "config",
            "params",
        ):
            return "Dict[str, Any]"
        elif name_lower.endswith(("_path", "_file", "_dir")):
            return "Union[str, Path]"
        else:
            return "Any"
