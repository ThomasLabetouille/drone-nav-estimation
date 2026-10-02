"""Step 4: realistic GNSS. Correlated position errors, then 150 ms of latency.

Six configurations of the step 3 filter on the 8-minute mission:
  A  white GNSS errors (the step 3 setting)
  B  correlated errors, filter treats them as white
  C  correlated errors, filter estimates them (18 states)
  then, with C's filter and 150 ms of latency:
  L1 latency ignored
  L2 latency compensated on the position (z += v * latency)
  L3 filter on a delayed horizon + output predictor

    python scripts/step4_gnss.py              # 50 runs per configuration, ~6 min
    python scripts/step4_gnss.py --runs 15    # quicker
"""

from __future__ import annotations

import argparse
import dataclasses
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
TURN_SHADE = "#e9e8e3"
DEG = 180 / np.pi

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

REAL = gnss.REALISTIC
NO_LATENCY = dataclasses.replace(REAL, latency_s=0.0)

# key: (label, gnss config, model the GNSS bias, latency mode)
CASES = {
    "A": ("A : erreurs blanches", gnss.GNSSConfig(), False, "ignore"),
    "B": ("B : erreurs corrélées, traitées comme blanches", NO_LATENCY, False, "ignore"),
    "C": ("C : erreurs corrélées, estimées (18 états)", NO_LATENCY, True, "ignore"),
    "L1": ("latence ignorée", REAL, True, "ignore"),
    "L2": ("latence compensée sur la position", REAL, True, "compensate"),
    "L3": ("horizon retardé + prédicteur de sortie", REAL, True, "delayed"),
}
COLORS = {"A": C1, "B": C2, "C": C3, "L1": C2, "L2": C3, "L3": C1}


def turn_intervals(tr, threshold_deg=2.0):
    turning = np.abs(np.degrees(tr.euler[:, 0])) > threshold_deg
    edges = np.flatnonzero(np.diff(turning.astype(int)))
    return [(tr.t[a + 1], tr.t[b + 1]) for a, b in zip(edges[::2], edges[1::2])]


def shade(ax, turns, label=False):
    for i, (a, b) in enumerate(turns):
        ax.axvspan(a, b, color=TURN_SHADE, lw=0, zorder=0,
                   label="virages" if (label and i == 0) else None)


def smooth(x, n=25):
    """Centred moving average; edges padded with the end values so the curve
    does not dive at both ends."""
    x = np.nan_to_num(x, nan=np.nanmean(x))
    xp = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")


def fig_correlated(logs, runs, path, r=0):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    ax = axes[0]
    lb, lc = logs["B"], logs["C"]
    t = lc.t
    sig = lc.sigma[r, :, 0]
    ax.fill_between(t, -3 * sig, 3 * sig, color=C3, alpha=0.13, lw=0, label="±3σ annoncé par C")
    ax.plot(lb.t, lb.err[r, :, 0], color=C2, lw=1.2, label="erreur de B")
    ax.plot(t, lc.err[r, :, 0], color=C3, lw=1.2, label="erreur de C")
    sig_b = lb.sigma[r, :, 0]
    ax.plot(lb.t, 3 * sig_b, color=C2, lw=0.8, ls="--", label="±3σ annoncé par B")
    ax.plot(lb.t, -3 * sig_b, color=C2, lw=0.8, ls="--")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Erreur de position nord [m]")
    ax.set_title("Un vol : erreur et incertitude annoncée")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    lo, hi = chi2.ppf([0.025, 0.975], 3 * runs) / (3 * runs)
    ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0, label="intervalle à 95 %")
    for key in ("A", "B", "C"):
        lg = logs[key]
        ax.plot(lg.t[1:], lg.nees["position"][:, 1:].mean(axis=0) / 3, color=COLORS[key], lw=1.1,
                label=key)
    ax.set_yscale("log")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Temps [s]")
    ax.set_title(f"NEES de position / ddl ({runs} runs)\nnécessite la vérité terrain")
    ax.legend(loc="center right", fontsize=8)

    ax = axes[2]
    for key in ("A", "B", "C"):
        lg = logs[key]
        ax.plot(lg.t[1:], smooth(lg.nis[:, 1:].mean(axis=0) / 6), color=COLORS[key], lw=1.3, label=key)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(0, 1.6)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Temps [s]")
    ax.set_title("NIS GNSS / ddl, lissé sur 5 s\nle seul test disponible en vol")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_latency(logs, turns, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True)
    for key in ("L1", "L2", "L3"):
        lg = logs[key]
        t = lg.t_out
        oe = lg.out_err
        pos = np.sqrt((oe[..., 0:2] ** 2).sum(axis=-1).mean(axis=0))
        yaw = np.sqrt((oe[..., 8] ** 2).mean(axis=0)) * DEG
        label = CASES[key][0]
        axes[0].plot(t, smooth(pos, 5), color=COLORS[key], lw=1.3, label=label)
        axes[1].plot(t, smooth(yaw, 5), color=COLORS[key], lw=1.3, label=label)
        axes[2].plot(lg.t[1:], smooth(lg.nis[:, 1:].mean(axis=0) / 6), color=COLORS[key], lw=1.3,
                     label=label)
    axes[0].set_ylabel("Position horizontale,\nRMS [m]")
    axes[0].set_title("150 ms de latence GNSS : l'état délivré au pilote automatique")
    axes[1].set_ylabel("Cap, RMS [°]")
    axes[2].set_ylabel("NIS GNSS / ddl\n(lissé sur 5 s)")
    axes[2].axhline(1.0, color=INK2, lw=0.8, ls="--")
    axes[2].set_xlabel("Temps [s]")
    for i, ax in enumerate(axes):
        shade(ax, turns, label=(i == 0))
        ax.set_xlim(0, logs["L3"].t_out[-1])
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(lg, t_min=120.0):
    m = lg.t > t_min
    oe = lg.out_err[:, m]
    return {
        "pos_h": float(np.sqrt((oe[..., 0:2] ** 2).sum(axis=-1).mean())),
        "pos_v": float(np.sqrt((oe[..., 2] ** 2).mean())),
        "vel": float(np.sqrt((oe[..., 3:5] ** 2).sum(axis=-1).mean())),
        "tilt": float(np.sqrt((oe[..., 6:8] ** 2).mean()) * DEG),
        "yaw": float(np.sqrt((oe[..., 8] ** 2).mean()) * DEG),
        "nees_pos": float(lg.nees["position"][:, m].mean() / 3),
        "nees_att": float(lg.nees["attitude"][:, m].mean() / 3),
        "nees_bg": float(lg.nees["gyro_bias"][:, m].mean() / 3),
        "nis": float(np.nanmean(lg.nis[:, m]) / 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    tr = trajectory3d.generate(trajectory3d.MISSION)
    imu_cfg, init = imu.IMUConfig(), fusion.InitConfig()
    logs, stats = {}, {}
    for key, (label, gcfg, model_bias, mode) in CASES.items():
        logs[key] = fusion.run(tr, imu_cfg, gcfg, init, args.runs, np.random.default_rng(args.seed),
                               model_gnss_bias=model_bias, latency_mode=mode)
        stats[key] = summary(logs[key])
        print(key, {k: round(v, 3) for k, v in stats[key].items()}, flush=True)

    turns = turn_intervals(tr)
    fig_correlated(logs, args.runs, img / "step4_correlated.png")
    fig_latency(logs, turns, img / "step4_latency.png")

    def row(key):
        s = stats[key]
        return (f"| {CASES[key][0]} | {s['pos_h']:.2f} | {s['pos_v']:.2f} | {s['vel']:.3f} "
                f"| {s['tilt']:.3f} | {s['yaw']:.2f} | {s['nees_pos']:.2f} | {s['nees_att']:.2f} "
                f"| {s['nees_bg']:.2f} | {s['nis']:.2f} |")

    header = ("| Configuration | Position horiz. [m] | Position vert. [m] | Vitesse horiz. [m/s] "
              "| Inclinaison [°] | Cap [°] | NEES pos. | NEES att. | NEES biais gyro | NIS |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# Étape 4 : résultats",
        "",
        f"Généré par `scripts/step4_gnss.py`. Mission de {tr.t[-1]:.0f} s, {args.runs} runs par "
        "configuration, même graine pour toutes. Erreurs RMS de l'état délivré en temps réel, "
        "NEES et NIS par degré de liberté (attendu : 1), sur t > 120 s.",
        "",
        f"GNSS réaliste : bruit blanc {REAL.pos_sigma_h} m / {REAL.pos_sigma_v} m, erreur corrélée "
        f"{REAL.corr_sigma_h} m / {REAL.corr_sigma_v} m (τ = {REAL.corr_tau:.0f} s), latence "
        f"{REAL.latency_s * 1000:.0f} ms.",
        "",
        "## Erreurs corrélées, sans latence",
        "",
        header, sep, row("A"), row("B"), row("C"),
        "",
        "## Latence de 150 ms (erreurs corrélées estimées)",
        "",
        header, sep, row("L1"), row("L2"), row("L3"),
        "",
    ]
    prov = provenance.build(__file__, params={"mission": trajectory3d.MISSION, "imu": imu_cfg,
                                              "init": init,
                                              "cases": {k: {"gnss": v[1], "model_gnss_bias": v[2],
                                                            "latency_mode": v[3]}
                                                        for k, v in CASES.items()}},
                            seed=args.seed, runs=args.runs)
    provenance.write(prov, ROOT / "docs" / "step4_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step4_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
