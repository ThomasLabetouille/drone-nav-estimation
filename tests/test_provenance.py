from pathlib import Path

from navsim import provenance


def test_fingerprint_ignores_line_endings(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    _, pa = provenance.code_fingerprint([a])
    _, pb = provenance.code_fingerprint([b])
    assert list(pa.values()) == list(pb.values())


def test_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\n")
    before, _ = provenance.code_fingerprint([f])
    f.write_text("a = 2\n")
    after, _ = provenance.code_fingerprint([f])
    assert before != after


def test_build_records_only_loaded_modules():
    import navsim.trajectory  # noqa: F401

    script = Path(provenance.__file__)
    prov = provenance.build(script, params={"k": 1}, seed=3, runs=2)
    assert "src/navsim/trajectory.py" in prov["files"]
    assert "src/navsim/provenance.py" in prov["files"]
    assert prov["seed"] == 3 and prov["params"] == {"k": 1}
    assert len(prov["code_fingerprint"]) == 64


def test_committed_results_match_current_code():
    """The results committed in docs/ must have been produced by the code
    currently in the repository."""
    import json

    root = Path(provenance.ROOT)
    for jp in sorted((root / "docs").glob("*_provenance.json")):
        prov = json.loads(jp.read_text(encoding="utf-8"))
        fp, _ = provenance.code_fingerprint([root / rel for rel in prov["files"]])
        assert fp == prov["code_fingerprint"], f"{jp.name} est périmé, relancer {prov['script']}"
