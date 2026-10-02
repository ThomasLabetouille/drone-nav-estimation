"""Step 3: IMU + GNSS error-state EKF (15 states) on the 8-minute mission.

    python scripts/step3_eskf.py              # 100 Monte-Carlo runs, ~3 min
    python scripts/step3_eskf.py --runs 20    # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim import fusion, gnss, imu, provenance, trajectory3d  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
AXIS_COLORS = (C1, C2, C3)
TURN_SHADE = "#e9e8e3"

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

DEG = 180 / np.pi
DOF = {"position": 3, "velocity": 3, "attitude": 3, "gyro_bias": 3, "accel_bias": 3}
BLOCK_LABELS = {"position": "Position", "velocity": "Vitesse", "attitude": "Attitude",
                "gyro_bias": "Biais gyro", "accel_bias": "Biais accéléro"}


def turn_intervals(tr, threshold_deg=2.0):
    # Padding with "not turning" at both ends closes a turn that is still
    # going on at the start or at the end of the flight (found by the linter:
    # without it, an unmatched start was silently dropped by zip).
    turning = np.abs(np.degrees(tr.euler[:, 0])) > threshold_deg
    edges = np.flatnonzero(np.diff(np.concatenate([[0], turning.astype(int), [0]])))
    last = len(tr.t) - 1
    return [(tr.t[a], tr.t[min(b, last)]) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def shade_turns(ax, turns, label=False):
    for i, (a, b) in enumerate(turns):
        ax.axvspan(a, b, color=TURN_SHADE, lw=0, zorder=0,
                   label="virages" if (label and i == 0) else None)


def fig_observability(tr, log, turns, path):
    t = log.t
    s = log.sigma.mean(axis=0)  # sigmas are nearly identical across runs
    fig, axes = plt.subplots(4, 1, figsize=(10, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 1, 1, 1]})
    ax = axes[0]
    ax.plot(tr.t, np.degrees(tr.euler[:, 0]), color=INK2, lw=1.2)
    ax.set_ylabel("Roulis [°]")
    ax.set_title("Ce que le filtre apprend, et quand : incertitude (1σ) moyenne sur les runs")

    ax = axes[1]
    ax.plot(t, s[:, 8] * DEG, color=C1, label="cap")
    ax.set_ylabel("Cap, 1σ [°]")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(t, s[:, 6] * DEG, color=C1, label="inclinaison autour du nord")
    ax.set_ylabel("Inclinaison, 1σ [°]")
    ax.legend(loc="upper right")
    axb = axes[3]
    axb.plot(t, s[:, 12], color=C2, label="biais accéléro x (avant)")
    axb.plot(t, s[:, 14], color=C3, label="biais accéléro z")
    axb.set_ylabel("Biais accéléro, 1σ [m/s²]")
    axb.legend(loc="upper right")
    axb.set_xlabel("Temps [s]")
    for ax in axes:
        shade_turns(ax, turns, label=(ax is axes[0]))
        ax.set_xlim(t[0], t[-1])
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gyro_bias(tr, log, turns, path):
    t = log.t
    s = log.sigma.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    shade_turns(ax, turns, label=True)
    for i, (name, c) in enumerate(zip(("x (roulis)", "y (tangage)", "z (lacet)"), AXIS_COLORS, strict=True)):
        ax.plot(t, s[:, 9 + i] * DEG, color=c, label=f"biais gyro {name}")
    ax.set_ylabel("1σ [°/s]")
    ax.set_xlabel("Temps [s]")
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Le biais du gyro de lacet ne s'apprend que pendant les virages")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_single_run(log, turns, path, r=0):
    t = log.t
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax = axes[0]
    for i, (name, c) in enumerate(zip("NED", AXIS_COLORS, strict=True)):
        ax.plot(t, log.err[r, :, i], color=c, lw=1.1, label=name)
    ax.set_ylabel("Erreur de position [m]")
    ax.set_title("Un vol : erreurs et estimations")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(-1.5, 1.5)

    ax = axes[1]
    sig = log.sigma[r, :, 8] * DEG
    ax.fill_between(t, -3 * sig, 3 * sig, color=C1, alpha=0.13, lw=0, label="±3σ du filtre")
    ax.plot(t, log.err[r, :, 8] * DEG, color=C1, label="erreur de cap")
    ax.set_ylabel("Cap [°]")
    ax.set_ylim(-16, 16)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax = axes[2]
    for i, c in enumerate(AXIS_COLORS):
        ax.plot(t, log.gyro_bias[r, :, i] * DEG, color=c, lw=1.0, ls=(0, (3, 2)))
        ax.plot(t, log.gyro_bias_est[r, :, i] * DEG, color=c, label="xyz"[i])
    ax.set_ylabel("Biais gyro [°/s]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="estimé\n(vrai en\npointillés)",
              title_fontsize=8)

    ax = axes[3]
    for i, c in enumerate(AXIS_COLORS):
        ax.plot(t, log.accel_bias[r, :, i], color=c, lw=1.0, ls=(0, (3, 2)))
        ax.plot(t, log.accel_bias_est[r, :, i], color=c, label="xyz"[i])
    ax.set_ylabel("Biais accéléro [m/s²]")
    ax.set_xlabel("Temps [s]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="estimé\n(vrai en\npointillés)",
              title_fontsize=8)
    for ax in axes:
        shade_turns(ax, turns)
        ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_consistency(log, runs, path):
    t = log.t[1:]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    panels = [(name, log.nees[name][:, 1:], DOF[name], BLOCK_LABELS[name]) for name in DOF]
    panels.append(("nis", log.nis[:, 1:], 6, "NIS GNSS"))
    for ax, (_name, values, dof, label) in zip(axes.flat, panels, strict=True):
        mean = values.mean(axis=0) / dof
        lo, hi = chi2.ppf([0.025, 0.975], runs * dof) / (runs * dof)
        ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0)
        ax.plot(t, mean, color=C1, lw=0.9)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        inside = np.mean((mean >= lo) & (mean <= hi))
        ax.set_title(f"{label} : {inside:.0%} dans l'intervalle", fontsize=10)
        ax.set_ylim(0, 2.5)
        ax.set_xlim(t[0], t[-1])
    for ax in axes[1]:
        ax.set_xlabel("Temps [s]")
    axes[0, 0].set_ylabel("NEES / ddl")
    axes[1, 0].set_ylabel("NEES ou NIS / ddl")
    fig.suptitle(f"Cohérence du filtre sur {runs} runs (attendu : 1, bande grise à 95 %)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sensitivity-runs", type=int, default=50)
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    tr = trajectory3d.generate(trajectory3d.MISSION)
    imu_cfg, gnss_cfg, init = imu.IMUConfig(), gnss.GNSSConfig(), fusion.InitConfig()
    log = fusion.run(tr, imu_cfg, gnss_cfg, init, args.runs, np.random.default_rng(args.seed))
    turns = turn_intervals(tr)

    fig_observability(tr, log, turns, img / "step3_observability.png")
    fig_gyro_bias(tr, log, turns, img / "step3_gyro_bias.png")
    fig_single_run(log, turns, img / "step3_single_run.png")
    fig_consistency(log, args.runs, img / "step3_consistency.png")

    t = log.t
    conv = t > 120.0
    e = log.err[:, conv]
    rms = np.sqrt((e**2).mean(axis=(0, 1)))
    s = log.sigma.mean(axis=0)

    def sig_at(tt, i, scale=1.0):
        return s[np.searchsorted(t, tt), i] * scale

    rows_nees = []
    for name, dof in DOF.items():
        mean = log.nees[name][:, 1:].mean(axis=0) / dof
        lo, hi = chi2.ppf([0.025, 0.975], args.runs * dof) / (args.runs * dof)
        rows_nees.append(f"| {BLOCK_LABELS[name]} | {mean.mean():.2f} | {np.mean((mean >= lo) & (mean <= hi)):.0%} |")
    nis = log.nis[:, 1:].mean(axis=0) / 6
    lo, hi = chi2.ppf([0.025, 0.975], args.runs * 6) / (args.runs * 6)
    rows_nees.append(f"| NIS GNSS | {nis.mean():.2f} | {np.mean((nis >= lo) & (nis <= hi)):.0%} |")

    first_turn = turns[0][0]

    # Sensitivity: attitude NEES before the first turn vs initial heading error.
    early = (t > 5.0) & (t < first_turn)
    late = t > 100.0
    rows_sens = []
    for yaw_sigma in (5.0, 2.0, 1.0):
        if yaw_sigma == init.yaw_sigma_deg and args.runs >= args.sensitivity_runs:
            lg = log
        else:
            lg = fusion.run(tr, imu_cfg, gnss_cfg, fusion.InitConfig(yaw_sigma_deg=yaw_sigma),
                            args.sensitivity_runs, np.random.default_rng(args.seed + 1))
        rows_sens.append(f"| {yaw_sigma:.0f}° | {lg.nees['attitude'][:, early].mean() / 3:.2f} "
                         f"| {lg.nees['attitude'][:, late].mean() / 3:.2f} |")
    lines = [
        "# Étape 3 : résultats",
        "",
        f"Généré par `scripts/step3_eskf.py`. Mission de {t[-1]:.0f} s, IMU à {1 / tr.dt:.0f} Hz, "
        f"GNSS à {gnss_cfg.rate_hz:.0f} Hz, {args.runs} runs. Premier virage à {first_turn:.0f} s.",
        "",
        "## Erreurs RMS après convergence (t > 120 s)",
        "",
        "| | Nord | Est | Bas |",
        "|---|---|---|---|",
        f"| Position [m] | {rms[0]:.2f} | {rms[1]:.2f} | {rms[2]:.2f} |",
        f"| Vitesse [m/s] | {rms[3]:.3f} | {rms[4]:.3f} | {rms[5]:.3f} |",
        "",
        "| | Inclinaison autour du nord | Inclinaison autour de l'est | Cap |",
        "|---|---|---|---|",
        f"| Attitude [°] | {rms[6] * DEG:.3f} | {rms[7] * DEG:.3f} | {rms[8] * DEG:.2f} |",
        "",
        "## Cohérence",
        "",
        "| Bloc | NEES ou NIS moyen / ddl (attendu : 1) | Temps dans l'intervalle à 95 % |",
        "|---|---|---|",
        *rows_nees,
        "",
        "## Cohérence de l'attitude selon l'erreur de cap initiale",
        "",
        "| σ du cap initial | NEES attitude / ddl, de 5 s au premier virage | après 100 s |",
        "|---|---|---|",
        *rows_sens,
        "",
        "## Observabilité : incertitude moyenne (1σ) avant et après les manœuvres",
        "",
        "| État | Avant le premier virage (t = 65 s) | Après (t = 90 s) | Fin de mission |",
        "|---|---|---|---|",
        f"| Cap [°] | {sig_at(65, 8, DEG):.2f} | {sig_at(90, 8, DEG):.2f} | {sig_at(t[-1], 8, DEG):.2f} |",
        f"| Inclinaison [°] | {sig_at(65, 6, DEG):.3f} | {sig_at(90, 6, DEG):.3f} | {sig_at(t[-1], 6, DEG):.3f} |",
        f"| Biais accéléro x [m/s²] | {sig_at(65, 12):.4f} | {sig_at(90, 12):.4f} | {sig_at(t[-1], 12):.4f} |",
        f"| Biais accéléro z [m/s²] | {sig_at(65, 14):.4f} | {sig_at(90, 14):.4f} | {sig_at(t[-1], 14):.4f} |",
        "",
        "| État | Avant les deux boucles (t = 105 s) | Après (t = 160 s) | Fin de mission |",
        "|---|---|---|---|",
        f"| Biais gyro z [°/s] | {sig_at(105, 11, DEG):.4f} | {sig_at(160, 11, DEG):.4f} | {sig_at(t[-1], 11, DEG):.4f} |",
        "",
    ]
    prov = provenance.build(__file__, params={"mission": trajectory3d.MISSION, "imu": imu_cfg,
                                              "gnss": gnss_cfg, "init": init},
                            seed=args.seed, runs=args.runs)
    provenance.write(prov, ROOT / "docs" / "step3_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step3_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
