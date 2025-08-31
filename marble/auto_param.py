from __future__ import annotations

import ast
import importlib.machinery
import importlib.abc
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_numeric_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _has_expose_decorator(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "expose_learnable_params":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "expose_learnable_params":
            return True
    return False


class _AutoParamTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.need_import = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if any(_is_numeric_constant(d) for d in node.args.defaults) and not _has_expose_decorator(node):
            node.decorator_list.append(ast.Name(id="expose_learnable_params", ctx=ast.Load()))
            self.need_import = True
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        if not self.need_import:
            return node
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "marble.wanderer":
                for alias in stmt.names:
                    if alias.name == "expose_learnable_params":
                        return node
        node.body.insert(
            0,
            ast.ImportFrom(
                module="marble.wanderer",
                names=[ast.alias(name="expose_learnable_params", asname=None)],
                level=0,
            ),
        )
        return node


class _AutoParamLoader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):  # type: ignore[override]
        tree = ast.parse(data, filename=path)
        transformer = _AutoParamTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        if transformer.need_import:
            data = ast.unparse(tree)
        else:
            data = data
        return super().source_to_code(data, path, _optimize=_optimize)


class _AutoParamFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[override]
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not spec or not isinstance(spec.loader, importlib.abc.SourceLoader):
            return spec
        try:
            origin = Path(spec.origin)
            if REPO_ROOT not in origin.resolve().parents:
                return spec
        except Exception:
            return spec
        spec.loader = _AutoParamLoader(fullname, spec.origin)
        return spec


@contextmanager
def enable_auto_param_learning(brain, learn_all: Optional[bool] = None):
    flag = brain.learn_all_numeric_parameters if learn_all is None else learn_all
    if not flag:
        yield
        return
    finder = _AutoParamFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
