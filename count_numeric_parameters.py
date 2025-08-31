#!/usr/bin/env python3
"""Count numeric parameters in function signatures across the repo.

This script parses all Python files in the repository and reports how many
function parameters with default numeric values exist. It also distinguishes
between parameters that are exposed as learnable via the
``@expose_learnable_params`` decorator and those that are not.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Tuple

# Only analyse the core ``marble`` package. Tests and other helper scripts are
# ignored so that counts reflect the library itself.
REPO_ROOT = Path(__file__).parent
TARGET_ROOT = REPO_ROOT / "marble"


def is_numeric_constant(node: ast.expr) -> bool:
    """Return True if *node* is an int or float constant."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def has_expose_decorator(node: ast.FunctionDef) -> bool:
    """Check if the function has an ``expose_learnable_params`` decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "expose_learnable_params":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "expose_learnable_params":
            return True
    return False


def count_file(path: Path) -> Tuple[int, int]:
    """Count numeric parameters in *path*.

    Returns a tuple ``(learnable, non_learnable)`` for this file.
    """
    with path.open("r", encoding="utf8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    learnable = 0
    non_learnable = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            numeric_defaults = [
                default
                for default in node.args.defaults
                if is_numeric_constant(default)
            ]
            if not numeric_defaults:
                continue

            if has_expose_decorator(node):
                learnable += len(numeric_defaults)
            else:
                non_learnable += len(numeric_defaults)
    return learnable, non_learnable


def main() -> None:
    learnable_total = 0
    non_learnable_total = 0

    for pyfile in TARGET_ROOT.rglob("*.py"):
        lrn, non = count_file(pyfile)
        learnable_total += lrn
        non_learnable_total += non

    total = learnable_total + non_learnable_total
    print(f"Total numeric parameters: {total}")
    print(f"Learnable numeric parameters: {learnable_total}")
    print(f"Non-learnable numeric parameters: {non_learnable_total}")


if __name__ == "__main__":
    main()
