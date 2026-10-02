"""Step 2: 3D fixed-wing trajectory, 6-axis IMU, strapdown navigation.

1. Generates an 8-minute mission and the matching ideal IMU increments.
2. Integrates the perfect IMU with three strapdown variants: the residual
   error is the algorithm's own error.
3. Monte-Carlo of unaided navigation with a MEMS IMU in straight and level
   flight, compared to the analytic error budget.

    python scripts/step2_strapdown.py              # 500 Monte-Carlo runs
    python scripts/step2_strapdown.py --runs 100   # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim import imu as imu_mod  # noqa: E402
from navsim import provenance  # noqa: E402
from navsim import strapdown, trajectory3d  # noqa: E402
from navsim.rotations import attitude_error  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9993"
GRID = "#e4e3de"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.6,
    "font.family": "DejaVu Sans",
})

METHOD_LABELS = {
    "naive": "naïve",
    "rotcomp": "+ compensation de rotation",
    "full": "+ coning et sculling",
}
METHOD_COLORS = {"naive": C2, "rotcomp": C3, "full": C1}

# Rough phase labels for the mission plot: (time [s], text)
PHASES = ((35, "montée", (-52, 0)), (130, "deux boucles", (10, 0)),
          (220, "pointe à 22 m/s", (10, 0)), (290, "virages en S", (12, 0)),
          (360, "descente", (10, 0)), (420, "boucle à gauche", (10, 0)))


def fig_trajectory(tr, path):
    t = tr.t
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1])

    ax = fig.add_subplot(gs[:, 0])
    ax.plot(tr.p_n[:, 1], tr.p_n[:, 0], color=C1)
    ax.plot(tr.p_n[0, 1], tr.p_n[0, 0], "o", color=INK, ms=6)
    ax.annotate("départ", (tr.p_n[0, 1], tr.p_n[0, 0]), xytext=(8, -4),
                textcoords="offset points", color=INK2, fontsize=8.5)
    ax.plot(tr.p_n[-1, 1], tr.p_n[-1, 0], "s", color=INK, ms=6)
    ax.annotate("arrivée", (tr.p_n[-1, 1], tr.p_n[-1, 0]), xytext=(8, -4),
                textcoords="offset points", color=INK2, fontsize=8.5)
    for tp, label, offset in PHASES:
        k = int(tp / tr.dt)
        ax.plot(tr.p_n[k, 1], tr.p_n[k, 0], "o", color=C1, ms=3.5)
        ax.annotate(label, (tr.p_n[k, 1], tr.p_n[k, 0]), xytext=offset,
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-900, 1100)
    ax.set_xlabel("Est [m]")
    ax.set_ylabel("Nord [m]")
    ax.set_title(f"Mission de {t[-1] / 60:.0f} min, vue de dessus")

    panels = (
        (-tr.p_n[:, 2], "Altitude [m]"),
        (np.degrees(tr.euler[:, 0]), "Roulis [°]"),
        (np.linalg.norm(tr.v_n, axis=1), "Vitesse [m/s]"),
    )
    for i, (y, label) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, y, color=C1)
        ax.set_ylabel(label)
        ax.set_xlim(t[0], t[-1])
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Temps [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_validation(tr, sols, path):
    t = tr.t[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, sol in sols.items():
        pe = np.linalg.norm(sol.p_n - tr.p_n, axis=1)[1:]
        axes[0].plot(t, np.where(pe > 1e-9, pe, np.nan), color=METHOD_COLORS[m],
                     label=METHOD_LABELS[m])
    # naive and rotcomp share the same attitude update: one curve for both
    for m, label in (("rotcomp", "sans correction de coning"), ("full", "avec correction de coning")):
        ae = np.degrees(np.linalg.norm(attitude_error(tr.q_nb, sols[m].q_nb), axis=1))[1:]
        axes[1].plot(t, np.where(ae > 1e-12, ae, np.nan), color=METHOD_COLORS[m], label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Erreur de position [m]")
    axes[0].set_title("IMU parfaite : erreur propre à l'algorithme")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Erreur d'attitude [°]")
    axes[1].set_title("Attitude : effet de la correction de coning")
    for ax in axes:
        ax.set_xlabel("Temps [s]")
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_dead_reckoning(t, rms_axis, rms_down, budget, runs, path):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    m = t >= 1.0
    half = 1.96 / np.sqrt(2 * runs)
    total = budget["total"]
    ax.fill_between(t[m], total[m] * (1 - half), total[m] * (1 + half), color=INK2, alpha=0.12,
                    lw=0, label=f"Intervalle à 95 % d'un Monte-Carlo de {runs} runs")
    # direct labels at the right end, pushed apart in log space so they never overlap
    ends = sorted(((curve[m][-1], name) for name, curve in budget.items() if name != "total"))
    placed = []
    for value, name in ends:
        y = np.log10(value)
        if placed and y - placed[-1] < 0.18:
            y = placed[-1] + 0.18
        placed.append(y)
        ax.plot(t[m], budget[name][m], color=MUTED, lw=1.0)
        ax.annotate(name, (t[m][-1], 10**y), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.plot(t[m], total[m], color=INK, lw=1.4, ls="--", label="Budget analytique (somme quadratique)")
    ax.plot(t[m], rms_axis[m], color=C1, lw=2.0, label=f"Monte-Carlo, {runs} vols")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, t[-1])
    ax.set_ylim(1e-3, None)
    ax.set_xlabel("Temps depuis la dernière position connue [s]")
    ax.set_ylabel("Erreur horizontale par axe, RMS [m]")
    k60 = np.searchsorted(t, 60.0)
    ax.set_title(f"IMU MEMS seule : {rms_axis[k60]:.0f} m d'erreur par axe après 60 s, "
                 "dominée par le biais gyro (pente t³)")
    ax.legend(loc="upper left")
    # leave room on the right for the direct labels
    ax.set_xlim(1.0, t[-1] * 3.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    # 1. Mission and ideal IMU
    tr = trajectory3d.generate(trajectory3d.MISSION)
    fig_trajectory(tr, img / "step2_trajectory.png")

    # 2. Perfect IMU through the three strapdown variants
    sols = {m: strapdown.integrate(tr.p_n[0], tr.v_n[0], tr.q_nb[0], tr.dtheta, tr.dvel, tr.dt, m)
            for m in strapdown.METHODS}
    fig_validation(tr, sols, img / "step2_strapdown_validation.png")
    val = {}
    for m, sol in sols.items():
        val[m] = (
            float(np.linalg.norm(sol.p_n[-1] - tr.p_n[-1])),
            float(np.degrees(np.linalg.norm(attitude_error(tr.q_nb, sol.q_nb), axis=1)).max()),
        )
        print(f"{m}: position {val[m][0]:.3g} m, attitude max {val[m][1]:.3g} deg")

    # 3. Monte-Carlo, MEMS IMU, straight and level flight
    cfg = imu_mod.IMUConfig()
    sl = trajectory3d.generate(trajectory3d.STRAIGHT_LEVEL)
    rng = np.random.default_rng(args.seed)
    meas = imu_mod.simulate(sl.dtheta, sl.dvel, cfg, rng, runs=args.runs)
    nav = strapdown.integrate(sl.p_n[0], sl.v_n[0], sl.q_nb[0], meas.dtheta, meas.dvel, sl.dt, "full")
    err = nav.p_n - sl.p_n
    rms_axis = np.sqrt(np.mean(err[..., 0] ** 2 + err[..., 1] ** 2, axis=0) / 2)
    rms_down = np.sqrt(np.mean(err[..., 2] ** 2, axis=0))
    budget = strapdown.error_budget(sl.t, cfg)
    fig_dead_reckoning(sl.t, rms_axis, rms_down, budget, args.runs, img / "step2_dead_reckoning.png")

    half = 1.96 / np.sqrt(2 * args.runs)
    rows = []
    for tt in (10, 30, 60, 120):
        k = int(round(tt / sl.dt))
        b = budget["total"][k]
        bv = np.sqrt((cfg.accel_bias0 * tt**2 / 2) ** 2 + cfg.accel_noise**2 * tt**3 / 3
                     + cfg.accel_bias_rw**2 * tt**5 / 20)
        rows.append(f"| {tt} s | {rms_axis[k]:.1f} | {b:.1f} | {rms_axis[k] / b:.2f} "
                    f"| {rms_down[k]:.1f} | {bv:.1f} |")
        print(rows[-1])

    lines = [
        "# Étape 2 : résultats",
        "",
        f"Généré par `scripts/step2_strapdown.py`. Mission de {tr.t[-1]:.0f} s, IMU à "
        f"{1 / tr.dt:.0f} Hz.",
        "",
        "## Strapdown avec une IMU parfaite",
        "",
        "| Variante | Erreur de position à la fin [m] | Erreur d'attitude max [°] |",
        "|---|---|---|",
    ]
    for m in strapdown.METHODS:
        lines.append(f"| {METHOD_LABELS[m]} | {val[m][0]:.3g} | {val[m][1]:.2g} |")
    lines += [
        "",
        f"## Navigation à l'estime, IMU MEMS, vol rectiligne ({args.runs} runs)",
        "",
        "| Temps | RMS horizontal par axe [m] | Budget [m] | Rapport | RMS vertical [m] "
        "| Budget vertical [m] |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"Avec {args.runs} runs, un rapport entre {1 - half:.2f} et {1 + half:.2f} est compatible "
        "avec le budget (intervalle à 95 % d'une RMS estimée).",
        "",
    ]
    prov = provenance.build(
        __file__, params={"mission": trajectory3d.MISSION, "straight_level": trajectory3d.STRAIGHT_LEVEL,
                          "imu": cfg, "imu_rate_hz": 1 / tr.dt, "oversample": 8},
        seed=args.seed, runs=args.runs,
    )
    provenance.write(prov, ROOT / "docs" / "step2_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step2_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
