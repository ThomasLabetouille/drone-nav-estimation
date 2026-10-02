"""The README is written by hand; the results files are generated. These tests
make sure the first does not drift away from the second.

Result tables in the README are marked with an HTML comment naming their
source, for instance <!-- source: docs/step3_results.md -->. Every number in
such a table (except the row label) must be found in that file, up to the
rounding shown in the README.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
README_NUMBER = re.compile(r"(?<![\w.,`])(\d+(?:,\d+)?)(?:·10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?(?![\w`])")
RESULT_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def marked_tables():
    """(source file, line number, table rows) for every marked table."""
    lines = README.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"<!-- source: (\S+) -->", line.strip())
        if not m:
            continue
        rows = []
        j = i + 1
        while j < len(lines) and not lines[j].startswith("|"):
            j += 1
        while j < len(lines) and lines[j].startswith("|"):
            rows.append((j + 1, lines[j]))
            j += 1
        out.append((m.group(1), i + 1, rows[2:]))  # skip header and separator
    return out


def readme_values(cell: str):
    cell = re.sub(r"`[^`]*`", "", cell)
    for mant, exp in README_NUMBER.findall(cell):
        decimals = len(mant.split(",")[1]) if "," in mant else 0
        value = float(mant.replace(",", "."))
        e = int(exp.translate(SUPERSCRIPT)) if exp else 0
        yield value, decimals, e


def matches(value, decimals, exp, candidates):
    for y in candidates:
        y_scaled = abs(y) / 10**exp
        if round(y_scaled, decimals) == round(value, decimals):
            return True
    return False


TABLES = marked_tables()


def test_readme_has_marked_tables():
    assert len(TABLES) >= 8


@pytest.mark.parametrize("source,line,rows", TABLES, ids=[f"README:{t[1]}" for t in TABLES])
def test_readme_table_matches_generated_results(source, line, rows):
    text = (ROOT / source).read_text(encoding="utf-8")
    candidates = [float(x) for x in RESULT_NUMBER.findall(text)]
    missing = []
    for lineno, row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        for cell in cells[1:]:
            for value, decimals, exp in readme_values(cell):
                if not matches(value, decimals, exp, candidates):
                    missing.append(f"ligne {lineno} : {cell}")
    assert not missing, f"absent de {source} : " + "; ".join(missing)


def test_readme_images_exist():
    for path in re.findall(r"\]\((docs/img/[^)]+)\)", README):
        assert (ROOT / path).exists(), path


def test_readme_file_tree_exists():
    block = README.split("## Organisation", 1)[1].split("```")[1]
    section = None
    for line in block.splitlines():
        if line and not line.startswith(" "):
            section = line.strip().rstrip("/")
        elif line.strip():
            name = line.split()[0]
            if section and not name.endswith("/") and "N" not in name:
                assert (ROOT / section / name).exists(), f"{section}/{name}"
