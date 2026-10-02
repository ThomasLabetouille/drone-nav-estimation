"""Provenance of a results file: which code, which parameters, which seed.

The fingerprint is computed by the script that writes the results, from the
source files as they are on disk at that moment. A fingerprint passed in by
the caller would only describe what the caller *thinks* it ran.

Line endings are normalised before hashing, so a Windows checkout (CRLF) and
a Linux checkout (LF) of the same commit give the same fingerprint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _normalised_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def code_fingerprint(paths) -> tuple[str, dict[str, str]]:
    """SHA-256 over the given files, sorted by path relative to the repo root.

    Returns (global fingerprint, {relative path: per-file sha256}).
    """
    files = sorted({Path(p).resolve() for p in paths})
    per_file = {}
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else f.name
        data = _normalised_bytes(f)
        per_file[rel] = hashlib.sha256(data).hexdigest()
        h.update(rel.encode() + b"\0" + data + b"\0")
    return h.hexdigest(), per_file


def loaded_sources(extra=()) -> list[Path]:
    """The navsim modules actually imported by the running script, plus any
    extra file (the script itself). Modules the script never imported do not
    affect its results, so they are left out of its fingerprint."""
    import sys

    pkg = (ROOT / "src" / "navsim").resolve()
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().parent == pkg:
            files.add(Path(f).resolve())
    return sorted(files) + [Path(p).resolve() for p in extra]


def _git_state() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        if rev.returncode != 0:
            return {"commit": None, "dirty": None}
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                               cwd=ROOT, capture_output=True, text=True, timeout=5)
        return {"commit": rev.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def build(script: str | Path, params: dict, seed: int, runs: int | None = None) -> dict:
    import matplotlib
    import numpy
    import scipy

    fingerprint, per_file = code_fingerprint(loaded_sources([script]))
    params_json = json.dumps(_jsonable(params), sort_keys=True)
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": Path(script).resolve().relative_to(ROOT).as_posix(),
        "code_fingerprint": fingerprint,
        "files": per_file,
        "git": _git_state(),
        "seed": seed,
        "runs": runs,
        "params": json.loads(params_json),
        "params_sha256": hashlib.sha256(params_json.encode()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def markdown(prov: dict) -> list[str]:
    git = prov["git"]
    if git["commit"]:
        git_txt = f"`{git['commit'][:12]}`" + (" (modifications non commitées)" if git["dirty"] else "")
    else:
        git_txt = "non disponible"
    return [
        "## Provenance",
        "",
        f"- Empreinte du code : `{prov['code_fingerprint'][:16]}` "
        f"({len(prov['files'])} fichiers, calculée par le script au moment de l'écriture)",
        f"- Commit git : {git_txt}",
        f"- Graine : {prov['seed']}" + (f", {prov['runs']} runs" if prov["runs"] else ""),
        f"- Paramètres : `{prov['params_sha256'][:16]}` (détail dans le fichier JSON voisin)",
        f"- Python {prov['versions']['python']}, numpy {prov['versions']['numpy']}, "
        f"scipy {prov['versions']['scipy']}",
        f"- Généré le {prov['generated_utc']}",
        "",
        "`python scripts/check_provenance.py` compare cette empreinte au code actuel.",
        "",
    ]


def write(prov: dict, json_path: Path):
    json_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
