"""Step 1: vertical channel, accelerometer + barometer Kalman filter.

Produces three figures in docs/img/ and a results table in
docs/step1_results.md.

    python scripts/step1_altitude.py            # 50 Monte-Carlo runs
    python scripts/step1_altitude.py --runs 10  # quicker
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

from navsim import provenance  # noqa: E402
from navsim.runner import dead_reckoning, simulate_and_run  # noqa: E402
from navsim.sensors import AccelConfig, BaroConfig  # noqa: E402
from navsim.trajectory import fixed_wing_vertical_profile  # noqa: E402

# --- plot style ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
TRUTH = "#52514e"
C_KF = "#2a78d6"  # blue: Kalman filter
C_BARO = "#eb6834"  # orange: barometer
C_DR = "#1baf7a"  # aqua: accelerometer only
SCEN_COLORS = {"A": C_KF, "B": C_BARO, "C": C_DR}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "font.family": "DejaVu Sans",
    }
)

# --- scenarios -------------------------------------------------------------
ACCEL = AccelConfig()
BARO = BaroConfig()
BARO_NO_DRIFT = BaroConfig(drift_sigma=0.0)

# name: (baro truth, baro as modelled by the filter, label)
SCENARIOS = {
    "A": (BARO_NO_DRIFT, BARO_NO_DRIFT, "A : baro sans dérive, filtre adapté"),
    "B": (BARO, BARO_NO_DRIFT, "B : baro avec dérive, non modélisée"),
    "C": (BARO, BARO, "C : baro avec dérive, modélisée dans l'état"),
}


def lowpass(x, dt, tau):
    """First-order low-pass, the usual way to smooth a baro vario."""
    y = np.empty_like(x)
    y[0] = x[0]
    alpha = dt / (tau + dt)
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


def baro_vario(baro, tau=1.0):
    dtb = baro.t[1] - baro.t[0]
    v = np.empty_like(baro.z)
    v[0] = 0.0
    v[1:] = np.diff(baro.z) / dtb
    return lowpass(v, dtb, tau)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# --- figures ---------------------------------------------------------------
def fig_estimation(profile, r, path):
    t = profile.t
    sig = np.sqrt(r.P[:, 0, 0])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})

    ax = axes[0]
    ax.plot(r.baro.t, r.baro.z, ".", ms=2.2, color=C_BARO, alpha=0.45,
            label="Baromètre (25 Hz)")
    ax.plot(t, profile.h, color=TRUTH, lw=1.2, ls="--", label="Vérité")
    ax.plot(t, r.x[:, 0], color=C_KF, label="Kalman (accéléro 100 Hz + baro)")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Altitude estimée sur une mission complète")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.fill_between(t, -3 * sig, 3 * sig, color=C_KF, alpha=0.13, lw=0,
                    label="±3σ annoncé par le filtre")
    ax.plot(t, r.x[:, 0] - profile.h, color=C_KF, label="Erreur du Kalman")
    ax.plot(r.baro.t, r.baro.drift, color=TRUTH, lw=1.2, ls="--",
            label="Dérive réelle du baro")
    ax.set_ylabel("Erreur altitude [m]")
    ax.set_title("Erreur d'altitude : le filtre suit la dérive du baro, qu'il ne peut pas observer")
    ax.legend(loc="upper left", ncol=3)

    ax = axes[2]
    sb = np.sqrt(r.P[:, 2, 2])
    ax.fill_between(t, r.x[:, 2] - 3 * sb, r.x[:, 2] + 3 * sb, color=C_KF,
                    alpha=0.13, lw=0, label="±3σ")
    ax.plot(t, r.accel.bias, color=TRUTH, lw=1.2, ls="--", label="Biais réel")
    ax.plot(t, r.x[:, 2], color=C_KF, label="Biais estimé")
    ax.set_ylabel("Biais accéléro [m/s²]")
    ax.set_xlabel("Temps [s]")
    ax.set_title("Biais de l'accéléromètre, estimé en ligne")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_why_fusion(profile, r, path, zoom=(65.0, 105.0)):
    t = profile.t
    h_dr, _ = dead_reckoning(profile, r.accel.a)
    vario = baro_vario(r.baro)

    fig = plt.figure(figsize=(10, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, h_dr - profile.h, color=C_DR, label="Accéléromètre seul (intégré deux fois)")
    ax.plot(t, r.x[:, 0] - profile.h, color=C_KF, label="Kalman accéléro + baro")
    ax.set_ylabel("Erreur altitude [m]")
    ax.set_xlabel("Temps [s]")
    ax.set_xlim(t[0], t[-1])
    ax.set_title(
        f"Accéléromètre seul : {abs(h_dr[-1] - profile.h[-1]):.0f} m d'erreur "
        f"après {t[-1]:.0f} s, à cause d'un biais de {abs(r.accel.bias[0]) * 1000 / 9.81:.1f} mg"
    )
    ax.legend(loc="lower left")

    m = (t >= zoom[0]) & (t <= zoom[1])
    mb = (r.baro.t >= zoom[0]) & (r.baro.t <= zoom[1])

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(r.baro.t[mb], r.baro.z[mb], ".", ms=3.5, color=C_BARO, alpha=0.6,
            label="Baromètre")
    ax.plot(t[m], r.x[m, 0], color=C_KF, lw=2.2, label="Kalman")
    ax.plot(t[m], profile.h[m], color=INK, lw=1.0, ls="--", label="Vérité")
    ax.set_title("Altitude (zoom, turbulence)")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Altitude [m]")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(r.baro.t[mb], vario[mb], color=C_BARO, lw=1.0, alpha=0.8,
            label="Vario baro (dérivée filtrée, τ = 1 s)")
    ax.plot(t[m], r.x[m, 1], color=C_KF, lw=2.2, label="Kalman")
    ax.plot(t[m], profile.v[m], color=INK, lw=1.0, ls="--", label="Vérité")
    ax.set_title("Vitesse verticale (zoom)")
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Vz [m/s]")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def block_mean(x, n):
    """Mean over consecutive, non-overlapping blocks of n samples."""
    m = len(x) // n
    return x[: m * n].reshape(m, n).mean(axis=1)


def fig_consistency(t, nees, nis_t, nis_single, n_runs, block, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    lo, hi = chi2.ppf([0.025, 0.975], 2 * n_runs) / n_runs
    ax = axes[0]
    ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0, label="Intervalle à 95 %")
    step = 25
    for name, curve in nees.items():
        ax.plot(t[::step], curve[::step], color=SCEN_COLORS[name], lw=1.4,
                label=SCENARIOS[name][2])
    ax.set_yscale("log")
    ax.set_xlabel("Temps [s]")
    ax.set_title(f"NEES moyen sur [h, v] ({n_runs} runs)\nnécessite la vérité terrain")
    ax.legend(loc="center right", fontsize=7.5)

    lo, hi = chi2.ppf([0.025, 0.975], block) / block
    ax = axes[1]
    ax.axhspan(lo, hi, color=INK2, alpha=0.12, lw=0, label="Intervalle à 95 %")
    tb = block_mean(nis_t[1:], block)
    for name, curve in nis_single.items():
        ax.step(tb, block_mean(curve[1:], block), where="mid",
                color=SCEN_COLORS[name], lw=1.6, label=f"Scénario {name}")
    ax.set_xlabel("Temps [s]")
    win = block * (nis_t[1] - nis_t[0])
    ax.set_title(f"NIS du baro sur un seul vol (fenêtres de {win:.0f} s)\n"
                 "le seul test disponible en vol réel")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)
    profile = fixed_wing_vertical_profile(seed=0)
    t = profile.t

    # Representative runs for the figures: realistic scenario (C) for the
    # estimation plot; no baro drift (A) to isolate the effect of fusion.
    rng = np.random.default_rng(args.seed)
    run_c = simulate_and_run(profile, ACCEL, BARO, ACCEL, BARO, rng)
    fig_estimation(profile, run_c, img / "step1_estimation.png")
    rng = np.random.default_rng(args.seed)
    run_a = simulate_and_run(profile, ACCEL, BARO_NO_DRIFT, ACCEL, BARO_NO_DRIFT, rng)
    fig_why_fusion(profile, run_a, img / "step1_why_fusion.png")

    block = int(round(10.0 * BARO.rate_hz))  # 10 s NIS windows
    nis_lo, nis_hi = chi2.ppf([0.025, 0.975], block) / block

    # Monte-Carlo
    nees, nis, nis_single, stats = {}, {}, {}, {}
    for name, (b_truth, b_model, _) in SCENARIOS.items():
        rng = np.random.default_rng(args.seed + 1)  # same noise draws per scenario
        nees_runs, nis_runs = [], []
        s = {k: [] for k in ("rmse_h", "rmse_v", "rmse_baro", "rmse_vario",
                             "dr_60", "dr_end")}
        for _ in range(args.runs):
            r = simulate_and_run(profile, ACCEL, b_truth, ACCEL, b_model, rng)
            nees_runs.append(r.nees_hv)
            nis_runs.append(r.nis)
            s["rmse_h"].append(rms(r.x[:, 0] - profile.h))
            s["rmse_v"].append(rms(r.x[:, 1] - profile.v))
            s["rmse_baro"].append(rms(r.baro.z - profile.h[r.baro.idx]))
            s["rmse_vario"].append(rms(baro_vario(r.baro) - profile.v[r.baro.idx]))
            if name == "A":
                h_dr, _ = dead_reckoning(profile, r.accel.a)
                i60 = np.searchsorted(t, 60.0)
                s["dr_60"].append(abs(h_dr[i60] - profile.h[i60]))
                s["dr_end"].append(abs(h_dr[-1] - profile.h[-1]))
        nees[name] = np.mean(nees_runs, axis=0)
        nis_arr = np.array(nis_runs)
        nis[name] = np.concatenate([[np.nan], nis_arr[:, 1:].mean(axis=0)])
        nis_single[name] = nis_arr[0]
        blocks = np.array([block_mean(x[1:], block) for x in nis_arr])
        stats[name] = {k: float(np.mean(v)) if v else float("nan") for k, v in s.items()}
        lo, hi = chi2.ppf([0.025, 0.975], 2 * args.runs) / args.runs
        stats[name]["nees_mean"] = float(np.mean(nees[name]))
        stats[name]["nees_in"] = float(np.mean((nees[name] >= lo) & (nees[name] <= hi)))
        stats[name]["nis_mean"] = float(np.nanmean(nis[name]))
        stats[name]["nis_alarm"] = float(np.mean(blocks > nis_hi))
        print(f"{name}: " + ", ".join(f"{k}={v:.3g}" for k, v in stats[name].items()))

    fig_consistency(t, nees, run_c.baro.t, nis_single, args.runs, block,
                    img / "step1_consistency.png")

    # Results table
    A, B, C = stats["A"], stats["B"], stats["C"]
    lines = [
        "# Étape 1 : résultats",
        "",
        f"Généré par `scripts/step1_altitude.py`, {args.runs} runs Monte-Carlo "
        f"par scénario, graine {args.seed}. Mission de {t[-1]:.0f} s.",
        "",
        "| | A | B | C |",
        "|---|---|---|---|",
        f"| Dérive baro dans la simulation | non | oui (σ = {BARO.drift_sigma} m) | oui |",
        "| Dérive baro dans le filtre | non | non | oui |",
        f"| RMSE altitude, baro brut [m] | {A['rmse_baro']:.2f} | {B['rmse_baro']:.2f} | {C['rmse_baro']:.2f} |",
        f"| RMSE altitude, Kalman [m] | {A['rmse_h']:.2f} | {B['rmse_h']:.2f} | {C['rmse_h']:.2f} |",
        f"| RMSE Vz, vario baro filtré [m/s] | {A['rmse_vario']:.2f} | {B['rmse_vario']:.2f} | {C['rmse_vario']:.2f} |",
        f"| RMSE Vz, Kalman [m/s] | {A['rmse_v']:.3f} | {B['rmse_v']:.3f} | {C['rmse_v']:.3f} |",
        f"| NEES moyen [h, v] (attendu : 2) | {A['nees_mean']:.2f} | {B['nees_mean']:.0f} | {C['nees_mean']:.2f} |",
        f"| Temps passé dans l'intervalle NEES 95 % | {A['nees_in']:.0%} | {B['nees_in']:.0%} | {C['nees_in']:.0%} |",
        f"| NIS moyen du baro (attendu : 1) | {A['nis_mean']:.2f} | {B['nis_mean']:.2f} | {C['nis_mean']:.2f} |",
        f"| Fenêtres de 10 s où le NIS dépasse son seuil à 97,5 % | {A['nis_alarm']:.1%} | {B['nis_alarm']:.1%} | {C['nis_alarm']:.1%} |",
        "",
        f"Accéléromètre seul, intégré depuis l'état vrai : erreur moyenne de "
        f"{A['dr_60']:.0f} m à 60 s et {A['dr_end']:.0f} m à {t[-1]:.0f} s.",
        "",
    ]
    prov = provenance.build(
        __file__,
        params={"accel": ACCEL, "baro": BARO,
                "scenarios": {k: {"baro_truth": v[0], "baro_model": v[1]} for k, v in SCENARIOS.items()},
                "nis_window_s": 10.0},
        seed=args.seed, runs=args.runs,
    )
    provenance.write(prov, ROOT / "docs" / "step1_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step1_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
