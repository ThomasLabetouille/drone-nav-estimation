"""Mutation testing: do the tests actually catch bugs?

Each mutation is a plausible bug (a flipped sign, a wrong frame, a wrong
unit convention...) written into a copy of the repository. The test suite is
then run on the copy. A mutation the tests catch is "killed"; one they do not
catch "survives". A test suite that has never been seen failing on a known
bug proves little; this is the check of the checks.

Some mutations are expected to survive, because they do not change the
results (equivalent mutants) or change them below any tolerance; each one is
labelled, and the report says why.

    python scripts/mutation_check.py              # all mutations, ~15 min
    python scripts/mutation_check.py --only M04   # one mutation

Writes docs/mutation_report.md and docs/mutation_provenance.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim.provenance import code_fingerprint  # noqa: E402

DESELECT = "tests/test_provenance.py::test_committed_results_match_current_code"


@dataclass(frozen=True)
class Mutation:
    id: str
    file: str
    old: str
    new: str
    bug: str            # what the mutation simulates
    tests: tuple        # test files run first (fast kill); the full suite follows if it survives
    expected: str = "killed"   # "killed" or "survives", with the reason in `note`
    note: str = ""


MUTATIONS = (
    # --- error-state EKF -----------------------------------------------------
    Mutation("M01", "src/navsim/eskf.py", "Phi[:, 3:6, 6:9] += -fx * dt", "Phi[:, 3:6, 6:9] += fx * dt",
             "signe du couplage vitesse/attitude dans F", ("tests/test_step3.py", "tests/test_oracles.py")),
    Mutation("M02", "src/navsim/eskf.py", "Phi[:, 6:9, 9:12] += -C * dt", "Phi[:, 6:9, 9:12] += C * dt",
             "signe du couplage attitude/biais gyro dans F", ("tests/test_step3.py", "tests/test_oracles.py")),
    Mutation("M03", "src/navsim/eskf.py", "qd[6:9] = imu_cfg.gyro_noise**2 * dt",
             "qd[6:9] = imu_cfg.gyro_noise**2 * dt * dt",
             "bruit de processus gyro en dt² au lieu de dt", ("tests/test_oracles.py", "tests/test_step3.py")),
    Mutation("M04", "src/navsim/eskf.py",
             "self.q = quat_normalize(quat_mul(quat_from_rotvec(dx[:, 6:9]), self.q))",
             "self.q = quat_normalize(quat_mul(self.q, quat_from_rotvec(dx[:, 6:9])))",
             "correction d'attitude injectée dans le repère avion au lieu de NED",
             ("tests/test_step3.py", "tests/test_step4.py")),
    Mutation("M05", "src/navsim/eskf.py", "return np.eye(3) + 0.5 * skew(dtheta_hat)",
             "return np.eye(3) - 0.5 * skew(dtheta_hat)",
             "signe de la jacobienne de réinitialisation (convention d'erreur locale au lieu de globale) : "
             "bug réel du projet jusqu'à l'étape 5", ("tests/test_oracles.py", "tests/test_step5.py")),
    Mutation("M06", "src/navsim/eskf.py",
             "P = IKH @ self.P @ np.swapaxes(IKH, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)",
             "P = IKH @ self.P", "forme de Joseph remplacée par la forme courte (I - KH) P",
             ("tests/test_oracles.py", "tests/test_properties.py"), expected="survives",
             note="mutant équivalent : avec le gain optimal, les deux formes sont égales en arithmétique "
                  "exacte. La forme de Joseph n'apporte que de la robustesse numérique"),
    Mutation("M07", "src/navsim/eskf.py", "self.H_gnss[0:3, s] = np.eye(3)", "self.H_gnss[0:3, s] = 0.0 * np.eye(3)",
             "biais GNSS oublié dans le modèle de mesure", ("tests/test_step4.py",)),
    # --- strapdown -----------------------------------------------------------
    Mutation("M08", "src/navsim/strapdown.py", "rot = dth + np.cross(prev_dth, dth) / 12.0",
             "rot = dth + np.cross(dth, prev_dth) / 12.0", "signe de la correction de coning",
             ("tests/test_step2.py",)),
    Mutation("M09", "src/navsim/strapdown.py", "np.array([0.0, 0.0, G * dt])", "np.array([0.0, 0.0, -G * dt])",
             "gravité de signe inversé en NED", ("tests/test_step2.py",)),
    Mutation("M10", "src/navsim/strapdown.py", "dv_b = dv + 0.5 * np.cross(dth, dv)",
             "dv_b = dv + 0.5 * np.cross(dv, dth)", "signe de la compensation de rotation",
             ("tests/test_step2.py",)),
    Mutation("M11", "src/navsim/strapdown.py", '"biais gyro": g * cfg.gyro_bias0 * t**3 / 6',
             '"biais gyro": g * cfg.gyro_bias0 * t**3 / 2', "erreur dans le budget d'erreur analytique",
             ("tests/test_oracles.py", "tests/test_step2.py")),
    # --- rotations -----------------------------------------------------------
    Mutation("M12", "src/navsim/rotations.py", "pw * qx + px * qw + py * qz - pz * qy",
             "pw * qx + px * qw - py * qz + pz * qy", "un terme du produit de quaternions",
             ("tests/test_oracles.py", "tests/test_step2.py")),
    Mutation("M13", "src/navsim/rotations.py", "cr * sp * cy + sr * cp * sy", "cr * sp * cy - sr * cp * sy",
             "convention des angles d'Euler", ("tests/test_oracles.py", "tests/test_step2.py")),
    # --- trajectory ----------------------------------------------------------
    Mutation("M14", "src/navsim/trajectory3d.py", "dpsi = G * np.tan(phi) / V", "dpsi = G * np.sin(phi) / V",
             "loi du virage coordonné", ("tests/test_step2.py", "tests/test_oracles.py")),
    Mutation("M15", "src/navsim/trajectory3d.py", "p = dphi - dpsi * np.sin(theta)",
             "p = dphi + dpsi * np.sin(theta)", "conversion vitesses d'Euler vers vitesses angulaires",
             ("tests/test_oracles.py", "tests/test_step2.py")),
    # --- sensors -------------------------------------------------------------
    Mutation("M16", "src/navsim/imu.py", "rng.normal(0.0, c.gyro_noise * sq, shape)",
             "rng.normal(0.0, c.gyro_noise * dt, shape)",
             "densité de bruit appliquée en dt au lieu de √dt", ("tests/test_oracles.py", "tests/test_step3.py")),
    Mutation("M17", "src/navsim/sensors.py", "q = cfg.drift_sigma * np.sqrt(1.0 - phi**2)",
             "q = cfg.drift_sigma * (1.0 - phi**2)", "discrétisation du processus de Gauss-Markov",
             ("tests/test_step1.py",)),
    Mutation("M18", "src/navsim/gnss.py", "self.phi = np.exp(-1.0 / (cfg.rate_hz * cfg.corr_tau))",
             "self.phi = np.exp(-1.0 / cfg.corr_tau)", "temps de corrélation GNSS en mauvaise unité",
             ("tests/test_step4.py",)),
    # --- step 1 filter -------------------------------------------------------
    Mutation("M19", "src/navsim/kf_altitude.py", "self.H[3] = 1.0", "self.H[3] = 0.0",
             "dérive baro absente du modèle de mesure", ("tests/test_step1.py", "tests/test_oracles.py")),
    Mutation("M20", "src/navsim/kf_altitude.py", "Qd = Phi @ E[:n, n:]", "Qd = E[:n, n:]",
             "formule de Van Loan incomplète", ("tests/test_oracles.py", "tests/test_step1.py")),
    Mutation("M21", "src/navsim/kf_altitude.py", "P[0, 3] = P[3, 0] = -sd2", "P[0, 3] = P[3, 0] = sd2",
             "signe du terme croisé de la covariance initiale (étape 1)", ("tests/test_step1.py",)),
    # --- fusion loop ---------------------------------------------------------
    Mutation("M22", "src/navsim/fusion.py", "z[:, 0:3] += ekf.v * gnss_cfg.latency_s",
             "z[:, 0:3] -= ekf.v * gnss_cfg.latency_s", "compensation de latence dans le mauvais sens",
             ("tests/test_oracles.py",)),
    Mutation("M23", "src/navsim/fusion.py", "P[0:3, 15:18] = P[15:18, 0:3] = -np.diag(corr2)",
             "P[0:3, 15:18] = P[15:18, 0:3] = np.diag(corr2)",
             "signe du terme croisé de la covariance initiale (étape 4)", ("tests/test_step4.py",)),
    Mutation("M24", "src/navsim/fusion.py", "            f = k + 1 - D\n", "            f = k - D\n",
             "décalage d'un échantillon IMU (5 ms) dans l'horizon retardé", ("tests/test_step4.py",
                                                                            "tests/test_oracles.py")),
    # --- provenance ----------------------------------------------------------
    Mutation("M25", "src/navsim/provenance.py", 'return path.read_bytes().replace(b"\\r\\n", b"\\n")',
             "return path.read_bytes()", "fins de ligne non normalisées avant le hachage",
             ("tests/test_provenance.py",)),
    # --- step 5: aiding sensors and wind -------------------------------------
    Mutation("M26", "src/navsim/eskf.py", "H[:, :, 6:9] = Ct @ skew(self.m_n)", "H[:, :, 6:9] = -Ct @ skew(self.m_n)",
             "signe de la jacobienne magnétomètre / attitude", ("tests/test_step5.py",)),
    Mutation("M27", "src/navsim/eskf.py", 'H[:, 0, self.blocks["wind"]] = -u[:, :2]',
             'H[:, 0, self.blocks["wind"]] = u[:, :2]', "signe du vent dans la mesure Pitot",
             ("tests/test_step5.py",)),
    Mutation("M28", "src/navsim/eskf.py",
             "d_vair = (Ct[:, 1, :] - beta[:, None] * v_air / V[:, None]) / V[:, None]",
             "d_vair = Ct[:, 1, :] / V[:, None]",
             "dérivée de 1/V oubliée dans la jacobienne du dérapage (erreur faite puis corrigée pendant "
             "l'écriture)", ("tests/test_step5.py",)),
    Mutation("M29", "src/navsim/eskf.py", "H[0, 2] = -1.0", "H[0, 2] = 1.0",
             "altitude baro prise égale à +p_D au lieu de -p_D", ("tests/test_step5.py",)),
    Mutation("M30", "src/navsim/fusion.py", "np.linalg.norm(ekf.omega, axis=1) > np.radians(thr)",
             "np.linalg.norm(ekf.omega, axis=1) < np.radians(thr)",
             "biais magnétomètre appris en ligne droite au lieu des virages", ("tests/test_step5.py",)),
    Mutation("M31", "src/navsim/trajectory3d.py", "    if np.any(dwind):\n        f_b = f_b + ",
             "    if False:\n        f_b = f_b + ",
             "accélération du vent absente de la force spécifique", ("tests/test_step5.py",)),
    Mutation("M32", "src/navsim/aiding.py", "self.cfg.bias_rw / np.sqrt(self.cfg.rate_hz)",
             "self.cfg.bias_rw / self.cfg.rate_hz", "marche aléatoire du biais magnétomètre en mauvaise unité",
             ("tests/test_step5.py",)),
    Mutation("M33", "src/navsim/aiding.py", "np.arctan2(-m_level[..., 1], m_level[..., 0])",
             "np.arctan2(m_level[..., 1], m_level[..., 0])", "signe du cap magnétique",
             ("tests/test_step5.py",)),
    Mutation("M34", "src/navsim/fusion.py", "J[:, :, 8] = -tas0[:, None] * u_perp", "J[:, :, 8] = tas0[:, None] * u_perp",
             "signe de la corrélation vent / cap dans la covariance initiale du vent", ("tests/test_step5.py",)),
    # --- step 6: faults and gating -------------------------------------------
    Mutation("M35", "src/navsim/eskf.py",
             "self.rejected = np.zeros(self.runs, dtype=bool) if gate is None else nis > gate",
             "self.rejected = np.zeros(self.runs, dtype=bool) if gate is None else nis < gate",
             "test d'innovation inversé", ("tests/test_step6.py",)),
    Mutation("M36", "src/navsim/eskf.py", "K[self.rejected] = 0.0", "K[self.rejected] = 1.0 * K[self.rejected]",
             "mesure rejetée mais fusionnée quand même", ("tests/test_step6.py",)),
    Mutation("M37", "src/navsim/fusion.py", '"gnss": 6, "baro": 1', '"gnss": 3, "baro": 1',
             "degrés de liberté du test GNSS (position seule au lieu de position + vitesse)",
             ("tests/test_step6.py",)),
    Mutation("M38", "src/navsim/fusion.py", "            if reset.any():\n                P, extra = wind_from_first_airspeed",
             "            if False:\n                P, extra = wind_from_first_airspeed",
             "protection contre le blocage du Pitot désactivée", ("tests/test_step6.py",)),
    Mutation("M39", "src/navsim/fusion.py",
             "np.where(dead_reckoning, aiding.wind.rw_without_gnss, aiding.wind.rw)",
             "np.where(dead_reckoning, aiding.wind.rw, aiding.wind.rw_without_gnss)",
             "marche aléatoire du vent réduite avec GNSS au lieu de sans", ("tests/test_step6.py",)),
    Mutation("M40", "src/navsim/fusion.py", "        if faults.gnss_lost(t):", "        if False:",
             "perte GNSS simulée ignorée", ("tests/test_step6.py",)),
    Mutation("M41", "src/navsim/aiding.py", "            field = field + disturbance_ned", "            field = field",
             "perturbation magnétique simulée ignorée", ("tests/test_step6.py",)),
)


def copy_repo(dst: Path):
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.egg-info", ".hypothesis")
    shutil.copytree(ROOT, dst, ignore=ignore)


def run_pytest(cwd: Path, targets, timeout=1800):
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--deselect", DESELECT,
           *targets]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    failed = [line.split(" ")[1] for line in out.splitlines() if line.startswith("FAILED ")]
    return proc.returncode, failed, out


def check_isolation(cwd: Path):
    """The copy must import its own (mutated) sources, not the original ones."""
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"))
    out = subprocess.run([sys.executable, "-c", "import navsim; print(navsim.__file__)"], cwd=cwd, env=env,
                         capture_output=True, text=True).stdout.strip()
    if not Path(out).resolve().is_relative_to(cwd.resolve()):
        raise RuntimeError(f"the mutated copy imports {out}, not its own sources")


def run_mutation(m: Mutation) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"mut_{m.id}_") as tmp:
        work = Path(tmp) / "repo"
        copy_repo(work)
        check_isolation(work)
        path = work / m.file
        src = path.read_text(encoding="utf-8")
        if src.count(m.old) != 1:
            return {"id": m.id, "status": "error", "detail": f"motif trouvé {src.count(m.old)} fois",
                    "seconds": 0.0}
        path.write_text(src.replace(m.old, m.new), encoding="utf-8")
        code, failed, out = run_pytest(work, list(m.tests))
        if code == 0:  # not caught by the targeted tests: run everything
            code, failed, out = run_pytest(work, ["tests"])
    status = "killed" if code != 0 else "survived"
    if code not in (0, 1):  # collection error, crash... still a detection
        failed = failed or [f"pytest code {code}"]
    return {"id": m.id, "status": status, "by": failed[0] if failed else "",
            "seconds": round(time.time() - t0, 1)}


def baseline_ok() -> bool:
    with tempfile.TemporaryDirectory(prefix="mut_base_") as tmp:
        work = Path(tmp) / "repo"
        copy_repo(work)
        check_isolation(work)
        code, failed, out = run_pytest(work, ["tests"])
        if code != 0:
            print(out[-3000:])
        return code == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    selected = [m for m in MUTATIONS if not args.only or m.id in args.only]
    if not args.skip_baseline:
        print("Suite non mutée...", flush=True)
        if not baseline_ok():
            print("La suite échoue sans mutation : corriger d'abord.")
            return 2
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = []
        for r in pool.map(run_mutation, selected):
            print(f"{r['id']}  {r['status']:9s} {r.get('by', '')}  ({r['seconds']} s)", flush=True)
            results.append(r)

    by_id = {m.id: m for m in MUTATIONS}
    unexpected = [r for r in results if r["status"] == "error"
                  or (r["status"] == "survived") != (by_id[r["id"]].expected == "survives")]
    if args.only:
        return 1 if unexpected else 0

    killed = sum(r["status"] == "killed" for r in results)
    lines = [
        "# Tests de mutation",
        "",
        "Généré par `scripts/mutation_check.py`. Chaque ligne est un bug plausible introduit dans une copie "
        "du dépôt, puis la suite de tests est lancée sur cette copie (sans le test de provenance, qui "
        "détecterait trivialement toute modification).",
        "",
        f"**{killed} mutations détectées sur {len(results)}.**",
        "",
        "| | Fichier | Bug simulé | Résultat | Détecté par |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        m = by_id[r["id"]]
        res = "détecté" if r["status"] == "killed" else ("survit" if r["status"] == "survived" else "erreur")
        if r["status"] == "survived" and m.expected == "survives":
            res = "survit (attendu)"
        elif r["status"] == "survived":
            res = "**survit, inattendu**"
        lines.append(f"| {m.id} | `{Path(m.file).name}` | {m.bug} | {res} | `{r.get('by', '')}` |")
    notes = [m for m in MUTATIONS if m.note and m.id in {r['id'] for r in results}]
    if notes:
        lines += ["", "## Mutations qui survivent, et pourquoi", ""]
        lines += [f"- **{m.id}** ({m.bug}) : {m.note}." for m in notes]
    lines.append("")

    files = sorted((ROOT / "src" / "navsim").glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + [
        Path(__file__).resolve()]
    fingerprint, per_file = code_fingerprint(files)
    prov = {"script": "scripts/mutation_check.py", "code_fingerprint": fingerprint, "files": per_file,
            "results": results}
    (ROOT / "docs" / "mutation_provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n",
                                                           encoding="utf-8")
    lines += [
        "## Provenance",
        "",
        f"- Empreinte des sources et des tests : `{fingerprint[:16]}` ({len(per_file)} fichiers)",
        "- Toute modification d'un module ou d'un test rend ce rapport périmé "
        "(`scripts/check_provenance.py`) : relancer le script.",
        "",
    ]
    (ROOT / "docs" / "mutation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
