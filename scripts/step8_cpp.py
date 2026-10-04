"""Step 8: the C++ port on the real logs of step 7, against the Python reference.

Same logs as step 7 (folder logs_px4/ next to the repository by default),
same configuration (R3), with and without the simulated GNSS outages.

    python scripts/step8_cpp.py               # ~5 min
    python scripts/step8_cpp.py --logs D:/logs
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from step7_replay import LOGS, sha256  # noqa: E402

from navsim import cpp_bridge, provenance, replay, ulog_reader  # noqa: E402
from navsim.rotations import quat_conj, quat_mul, rotvec_from_quat  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
COLORS = {"A1": "#2a78d6", "A2": "#eb6834", "B": "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.2,
    "font.family": "DejaVu Sans",
})


def compare(py, cc):
    datt = np.linalg.norm(rotvec_from_quat(quat_mul(py.q, quat_conj(cc["q"]))), axis=1)
    dec = {name: int(np.sum(py.nis[name][2] != cc["nis"][name][2])) for name in py.nis}
    n_upd = sum(len(v[0]) for v in py.nis.values())
    return {
        "epochs": len(py.t), "same_epochs": bool(np.array_equal(py.t, cc["t"])),
        "dp": float(np.abs(py.p - cc["p"]).max()), "dv": float(np.abs(py.v - cc["v"]).max()),
        "datt": float(datt.max()), "dextra": float(np.abs(py.extra - cc["extra"]).max()),
        "dsigma": float((np.abs(py.sigma - cc["sigma"]) / py.sigma).max()),
        "updates": n_upd, "decisions_differ": sum(dec.values()),
        "dp_t": np.linalg.norm(py.p - cc["p"], axis=1), "t": py.t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=ROOT.parent / "logs_px4")
    args = ap.parse_args()
    img = ROOT / "docs" / "img"

    build = cpp_bridge.build()
    unit = subprocess.run([str(build / "test_eskf")], capture_output=True, text=True, check=True).stdout
    timing_line = [ln for ln in unit.splitlines() if ln.startswith("predict")][0]
    print(unit)

    rows, hashes, curves = [], {}, {}
    for lid, (name, expected) in LOGS.items():
        path = args.logs / f"{lid}.ulg"
        if not path.exists():
            sys.exit(f"{path} manquant : télécharger https://review.px4.io/download?log={lid}")
        hashes[name] = sha256(path)
        if hashes[name] != expected:
            print(f"attention : {path.name} n'est pas le fichier des résultats publiés ({hashes[name][:12]})")
        data = ulog_reader.load(path, name)
        cfg = replay.config_like_ekf2(data)
        for label, c in (("vol complet", cfg),
                         ("pertes GNSS de 30 s", dataclasses.replace(
                             cfg, gnss_outage=replay.outage_windows(data.info["airborne"])))):
            t0 = time.perf_counter()
            py = replay.run(data, c)
            t_py = time.perf_counter() - t0
            t0 = time.perf_counter()
            cc = cpp_bridge.run(data, c, build)
            t_cc = time.perf_counter() - t0
            r = compare(py, cc)
            r.update(name=name, label=label, t_py=t_py, t_cc=t_cc, t_filter=cc["stats"]["filter_seconds"],
                     us=cc["stats"]["us_per_event"], events=int(cc["stats"]["events"]),
                     flight=float(data.t_imu[-1] - py.t[0]))
            rows.append(r)
            if label == "vol complet":
                curves[name] = (r["t"], r["dp_t"])
            print(name, label, {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, flush=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, (t, d) in curves.items():
        ax.plot(t - t[0], np.maximum(d, 1e-16), color=COLORS[name], label=f"vol {name}")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-6)
    ax.set_xlabel("Temps depuis le début du rejeu [s]")
    ax.set_ylabel("|position C++ − Python| [m]")
    ax.set_title("Écart entre le portage C++ et la référence Python, sur les vols réels")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(img / "step8_agreement.png", dpi=150)
    plt.close(fig)

    lines = [
        "# Étape 8 : résultats",
        "",
        "Généré par `scripts/step8_cpp.py`. Configuration R3 de l'étape 7, mêmes logs. Écarts maximaux sur "
        "toutes les époques GNSS du vol.",
        "",
        "## Concordance",
        "",
        "| Vol | Rejeu | Époques | Position [m] | Vitesse [m/s] | Attitude [rad] | États ajoutés | σ (relatif) "
        "| Mises à jour | Décisions du test différentes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['label']} | {r['epochs']} | {r['dp']:.1e} | {r['dv']:.1e} | {r['datt']:.1e} "
                     f"| {r['dextra']:.1e} | {r['dsigma']:.1e} | {r['updates']} | {r['decisions_differ']} |")
    lines += [
        "",
        "## Temps de calcul (cette machine)",
        "",
        "| Vol | Rejeu | Durée rejouée [s] | Événements | Python [s] | C++, programme complet [s] | C++, filtre seul [s] "
        "| C++, par événement [µs] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['label']} | {r['flight']:.0f} | {r['events']} | {r['t_py']:.1f} "
                     f"| {r['t_cc']:.2f} | {r['t_filter']:.2f} | {r['us']:.2f} |")
    lines += ["", f"Tests unitaires C++ (`cpp/tests/test_eskf.cpp`) : {timing_line}.", ""]

    cpp_sources = [p for p in sorted((ROOT / "cpp").rglob("*")) if p.suffix in (".cpp", ".hpp")
                   or p.name == "CMakeLists.txt"]
    cpp_sources = [p for p in cpp_sources if "build" not in p.relative_to(ROOT / "cpp").parts]
    prov = provenance.build(__file__, params={
        "logs": {name: {"review_px4_id": lid, "sha256": hashes[name]} for lid, (name, _) in LOGS.items()},
    }, seed=0, runs=1)
    # the results also depend on the C++ sources and on the step 7 script it imports
    prov["code_fingerprint"], prov["files"] = provenance.code_fingerprint(
        provenance.loaded_sources([__file__, ROOT / "scripts" / "step7_replay.py", *cpp_sources]))
    provenance.write(prov, ROOT / "docs" / "step8_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step8_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
