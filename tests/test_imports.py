"""Smoke tests: does every module import, and does every script's `from NSM... import x`
still point at something that exists?

Cheap, but catches the most common breakage after a refactor: a function moved or
renamed in one module while another still imports it from the old place.
"""
import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PACKAGES = ("NSM", "hierarchy", "encoder")
LOCAL_TOP_LEVEL = set(LIBRARY_PACKAGES) | {p.stem for p in REPO_ROOT.glob("*.py")}

# Need 3D Slicer's Python (slicer, qt, ctk, tiny3d); tested only inside Slicer.
SLICER_ONLY = {"SurfaceToolbox.py", "shape_completion_slicer.py"}
SKIP_DIRS = {"tests", "deprecated", "FossilNsmModule", ".git", ".github"}

KNOWN_BROKEN = {}


def _known_broken(key):
    if key in KNOWN_BROKEN:
        return [pytest.mark.xfail(strict=True, reason="KNOWN BUG: " + KNOWN_BROKEN[key])]
    return []


def _library_modules():
    names = []
    for package in LIBRARY_PACKAGES:
        names.append(package)
        for info in pkgutil.walk_packages([str(REPO_ROOT / package)], package + "."):
            if not set(info.name.split(".")) & SKIP_DIRS:
                names.append(info.name)
    return sorted(names)


def _python_files():
    files = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if set(rel.parts) & SKIP_DIRS or rel.name in SLICER_ONLY or rel.name == "setup.py":
            continue
        files.append(rel.as_posix())
    return files


@pytest.mark.parametrize(
    "module", [pytest.param(m, marks=_known_broken(m)) for m in _library_modules()])
def test_library_module_imports(module):
    try:
        importlib.import_module(module)
    except SystemExit:
        # e.g. hierarchy/visualize.py parses sys.argv at import time; it's a script, and
        # its own imports are still checked by test_script_local_imports_resolve.
        pytest.skip(f"{module} runs as a script at import time")


@pytest.mark.parametrize(
    "rel_path", [pytest.param(f, marks=_known_broken(f)) for f in _python_files()])
def test_script_local_imports_resolve(rel_path):
    """Parse (don't run) each script and check its imports from this repo resolve.
    Scripts without an `if __name__ == "__main__"` guard would start running if imported."""
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"), filename=rel_path)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] not in LOCAL_TOP_LEVEL:
                continue
            try:
                module = importlib.import_module(node.module)
            except SystemExit:
                continue  # an unguarded script that exits on import (e.g. argparse)
            except Exception as error:
                problems.append(f"line {node.lineno}: import {node.module} failed: {error!r}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    try:  # `from pkg import submodule`
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        problems.append(f"line {node.lineno}: {node.module} has no '{alias.name}'")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in LIBRARY_PACKAGES:
                    try:
                        importlib.import_module(alias.name)
                    except Exception as error:
                        problems.append(f"line {node.lineno}: import {alias.name} failed: {error!r}")
    assert not problems, f"{rel_path}:\n  " + "\n  ".join(problems)
