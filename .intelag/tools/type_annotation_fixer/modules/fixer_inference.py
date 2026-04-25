"""
╭────────────────────────────────────────────────────────────────────────────────────────╮
   ⚡ INTELAG PROPRIETARY CODE
   🏢 © 2025 Intelag LLC • All Rights Reserved
   📋 Licensed as Proprietary and Confidential
   ⚠️ Unauthorized use, reproduction, or distribution is strictly prohibited
   👤 Intelag LLC <intelag@outlook.com>
   🕒 Generated on: 2025-09-17 10:32:11
╰────────────────────────────────────────────────────────────────────────────────────────╯

Type inference functionality for the type annotation fixer.
"""

import ast
from typing import Optional, Set


class TypeInferenceMixin:
    """
    Mixin class for type inference functionality.
    """

    def infer_type_from_context(
        self, node: ast.AST, source_lines: list[str], type_mappings: dict[str, str]
    ) -> Optional[str]:
        """
        Infer type annotation from AST node and context.

        Args:
            node: AST node (function def, variable assignment, etc.)
            source_lines: Source code lines for context
            type_mappings: Dictionary of type mappings for inference

        Returns:
            Suggested type annotation string
        """
        if isinstance(node, ast.FunctionDef):
            return self._infer_function_return_type(node, source_lines, type_mappings)
        elif isinstance(node, ast.arg):
            return self._infer_parameter_type(node, source_lines, type_mappings)

        return None

    def _infer_function_return_type(
        self,
        func_node: ast.FunctionDef,
        _source_lines: list[str],
        type_mappings: dict[str, str],
    ) -> str:
        """Infer return type from function body."""
        # Look for return statements
        return_types: Set[str] = set()

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value is None:
                    return_types.add("None")
                else:
                    inf = self._infer_value_type(node.value)
                    if inf:
                        return_types.add(inf)

        # Check function name patterns for common conventions
        func_name = func_node.name.lower()
        if func_name.startswith(("is_", "has_", "can_", "should_", "exists")):
            return "bool"
        elif func_name.startswith("get_") and "list" in func_name:
            return "List[Any]"
        elif func_name.startswith("get_") and "dict" in func_name:
            return "Dict[str, Any]"
        elif func_name in ("__init__", "setup", "teardown"):
            return "None"

        # If multiple return types, use Union or Any
        if len(return_types) == 0:
            return "None"  # No explicit returns found
        elif len(return_types) == 1:
            ret_type: str = return_types.pop()
            return str(type_mappings.get(ret_type, ret_type) or "Any")
        elif "None" in return_types and len(return_types) == 2:
            other_type_set = return_types - {"None"}
            if not other_type_set:
                return "None"
            other_type: str = other_type_set.pop()
            mapped_type: str = type_mappings.get(other_type, other_type)
            return f"Optional[{mapped_type}]"
        else:
            return "Any"

    def _infer_value_type(self, value_node: Optional[ast.AST]) -> Optional[str]:
        """Infer type from a value node (literal, call, etc.)."""
        if value_node is None:
            return None
        if isinstance(value_node, ast.Constant):
            return type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            if not value_node.elts:
                return "List[Any]"
            inner = self._infer_value_type(value_node.elts[0])
            return f"List[{inner}]" if inner else "List[Any]"
        elif isinstance(value_node, ast.Dict):
            if not value_node.keys:
                return "Dict[str, Any]"
            k = self._infer_value_type(value_node.keys[0]) or "str"
            v = self._infer_value_type(value_node.values[0]) or "Any"
            return f"Dict[{k}, {v}]"
        elif isinstance(value_node, ast.Set):
            if not value_node.elts:
                return "Set[Any]"
            inner = self._infer_value_type(value_node.elts[0])
            return f"Set[{inner}]" if inner else "Set[Any]"
        elif isinstance(value_node, ast.Tuple):
            return "Tuple[Any, ...]"
        elif isinstance(value_node, ast.Name):
            var_name = value_node.id.lower()
            if var_name in ("true", "false"):
                return "bool"
            if "list" in var_name:
                return "List[Any]"
            if "dict" in var_name:
                return "Dict[str, Any]"
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id.lower()
                if func_name in ("str", "int", "float", "bool", "list", "dict", "set"):
                    return func_name
        return None

    def _infer_parameter_type(
        self,
        param_node: ast.arg,
        _source_lines: list[str],
        _type_mappings: dict[str, str],
    ) -> str:
        """Infer parameter type from name and context."""
        param_name = param_node.arg.lower()

        # Common parameter name patterns
        if param_name in ("self", "cls"):
            return ""  # Don't annotate self/cls
        elif param_name.endswith(("_id", "id")) or param_name in (
            "count",
            "size",
            "length",
        ):
            return "int"
        elif param_name.endswith(("_name", "name", "title", "text", "message", "_str")):
            return "str"
        elif (
            param_name.endswith(("_flag", "_enabled"))
            or param_name.startswith("is_")
            or param_name in ("flag", "enable", "success")
        ):
            return "bool"
        elif param_name.endswith(("_list", "_items", "indices")) or param_name in (
            "items",
            "values",
        ):
            return "List[Any]"
        elif param_name.endswith(
            ("_dict", "_data", "_config", "_params")
        ) or param_name in ("data", "config", "params", "payload"):
            return "Dict[str, Any]"
        elif param_name.endswith(("_path", "_file", "_dir")) or param_name in (
            "path",
            "filepath",
            "filename",
        ):
            return "Union[str, Path]"
        elif param_name.endswith(("_idx", "_index")):
            return "int"

        return "Any"


def example_usage() -> None:
    """
    Example usage of the TypeInferenceMixin class.
    """

    # Create a sample function AST node
    source_code = """
def greet(name):
    return f"Hello, {name}!"

def get_items(indices):
    return [str(i) for i in indices]

def check_status(is_enabled, count):
    if not is_enabled: return False
    return count > 0

async def fetch_data(url_path):
    return {"status": "ok", "data": []}
"""

    tree = ast.parse(source_code)
    type_mappings: dict[str, str] = {
        "str": "str",
        "int": "int",
        "bool": "bool",
        "list": "List[Any]",
        "dict": "Dict[str, Any]",
    }

    # Create an instance of the mixin (normally this would be mixed into another class)
    class TestInference(TypeInferenceMixin):
        pass

    inferrer: TestInference = TestInference()
    source_lines: list[str] = list(source_code.splitlines())

    print("Type inference examples:")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inferred_type = inferrer.infer_type_from_context(
                node, source_lines, type_mappings
            )
            print(f"Function '{node.name}': inferred return type = {inferred_type}")
        elif isinstance(node, ast.arg):
            inferred_type = inferrer.infer_type_from_context(
                node, source_lines, type_mappings
            )
            if inferred_type:
                print(f"Parameter '{node.arg}': inferred type = {inferred_type}")


if __name__ == "__main__":
    example_usage()
