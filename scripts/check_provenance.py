"""Compare the code fingerprint recorded with each results file to the code
currently on disk.

    python scripts/check_provenance.py

Exit code 0 if every results file matches the current code, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim.provenance import code_fingerprint  # noqa: E402


def check(json_path: Path) -> bool:
    prov = json.loads(json_path.read_text(encoding="utf-8"))
    recorded = prov["files"]
    missing = [rel for rel in recorded if not (ROOT / rel).exists()]
    current_paths = [ROOT / rel for rel in recorded if (ROOT / rel).exists()]
    fingerprint, current = code_fingerprint(current_paths)

    name = json_path.name
    if not missing and fingerprint == prov["code_fingerprint"]:
        print(f"OK      {name}  ({fingerprint[:16]})")
        return True
    print(f"PERIME  {name}  enregistré {prov['code_fingerprint'][:16]}, actuel {fingerprint[:16]}")
    for rel in missing:
        print(f"        fichier disparu : {rel}")
    for rel, sha in recorded.items():
        if rel in current and current[rel] != sha:
            print(f"        modifié depuis : {rel}")
    return False


def main() -> int:
    files = sorted((ROOT / "docs").glob("*_provenance.json"))
    if not files:
        print("Aucun fichier de provenance dans docs/")
        return 1
    results = [check(f) for f in files]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
