"""Stage previously retrieved, hash-qualified public benchmark files.

This script never changes source bytes. It copies the official and paper source
files from the pre-existing source audit into the campaign raw-source folder.
Unretrieved public sources stay explicitly unavailable rather than being
replaced by hand-entered values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "validation/source_matrix"
OUTPUT = ROOT / "docs/passive_hull_validation/raw_sources"
EXPECTED = json.loads((AUDIT / "admitted_reference.json").read_text())["archived_sources"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename, metadata in EXPECTED.items():
        source = AUDIT / filename
        if not source.is_file():
            raise SystemExit(f"Previously qualified source is missing: {source}")
        actual = sha256(source)
        if actual != metadata["sha256"]:
            raise SystemExit(f"Source integrity failure for {filename}: {actual}")
        destination = OUTPUT / filename
        if destination.exists() and sha256(destination) != actual:
            raise SystemExit(f"Refusing to replace modified raw source: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        copied.append({"filename": filename, "sha256": actual,
                       "bytes": destination.stat().st_size,
                       "source_identifier": "NMRI/ATMA archived source matrix"})
    (OUTPUT / "retrieval_manifest.json").write_text(json.dumps({
        "retrieval_date": "2026-09-28",
        "method": "byte-for-byte copy from existing provenance-audited acquisition",
        "files": copied,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Staged and verified {len(copied)} source files in {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
