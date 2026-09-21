"""Fail if production code imports the legacy simulator directly or dynamically."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "bcod_sim"
LEGACY_ROOTS = ("legacy", "sim_v3", "simv3", "icra_simulator")


def violations(source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            )
            if not dynamic or not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            names = [node.args[0].value] if isinstance(node.args[0].value, str) else []
        else:
            continue
        for name in names:
            if name.split(".", 1)[0].lower() in LEGACY_ROOTS:
                found.append(f"{filename}:{node.lineno}: forbidden legacy import {name}")
    return found


def main() -> int:
    if not PACKAGE.is_dir():
        print(f"Missing production package: {PACKAGE}", file=sys.stderr)
        return 1
    failures: list[str] = []
    files = sorted(PACKAGE.rglob("*.py"))
    if not files:
        print("Production package has no Python files", file=sys.stderr)
        return 1
    for path in files:
        failures.extend(violations(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"PASS: {len(files)} production Python file(s) have no legacy imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
