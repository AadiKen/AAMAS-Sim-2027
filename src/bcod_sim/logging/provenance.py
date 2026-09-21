"""Compute software and runtime provenance from observed state."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess

import torch

def _run_git(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def software_provenance(root: str | Path) -> dict[str, object]:
    root = Path(root)
    commit = _run_git(root, ["rev-parse", "HEAD"]) or "UNBORN"
    status = _run_git(root, ["status", "--porcelain", "--untracked-files=all"])
    declaration_file = root/"pyproject.toml"
    dependency_file = root/"requirements.lock"
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "onnx", "onnxruntime", "pyarrow", "pydantic", "PyYAML", "torch")}
    tree = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        tree.update(path.relative_to(root).as_posix().encode()+b"\0"+path.read_bytes()+b"\0")
    tree.update(b"pyproject.toml\0"+declaration_file.read_bytes())
    tree.update(b"requirements.lock\0"+dependency_file.read_bytes())
    source_tree_hash = tree.hexdigest()
    dependency_hash = hashlib.sha256(dependency_file.read_bytes()).hexdigest()
    return {"git_commit": commit, "dirty_patch_sha256": source_tree_hash if status or commit == "UNBORN" else None,
            "source_tree_sha256": source_tree_hash,
            "dependency_file": dependency_file.name,
            "dependency_lock_sha256": dependency_hash,
            "packages": packages}


def runtime_provenance(dtype: torch.dtype, device: torch.device) -> dict[str, str]:
    return {"backend": "torch", "torch_version": torch.__version__, "device": str(device),
            "dtype": str(dtype).removeprefix("torch.")}
