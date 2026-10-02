"""Step 5: magnetometer, barometer, Pitot tube and wind, in a wind of 5 m/s.

All configurations use the step 4 filter (realistic GNSS, estimated GNSS
bias, delayed horizon) on the mission flown in wind, with the same random
draws for the IMU and the GNSS:
  E1  GNSS only, initial heading from the GNSS course
  E2  + magnetometer (initial heading, 3-axis fusion, hard-iron bias)
  E2b same, but the bias is learnt all the time (see "Écarts")
  E3  + barometer (with its drift state)
  E4  + Pitot tube and wind states
  E5a + zero-sideslip pseudo-measurement, sigma 2 deg at 10 Hz (naive)
  E5i same, sigma 6 deg, but the wind starts at zero instead of the first Pitot sample
  E5  sigma 6 deg (the sideslip error is correlated in turns), wind from the first Pitot sample

    python scripts/step5_aiding.py              # 40 runs per configuration, ~15 min
    python scripts/step5_aiding.py --runs 10    # quicker
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim import aiding, fusion, gnss, imu, provenance, trajectory3d

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
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

MAG = aiding.MagConfig()
MAG_ALWAYS = dataclasses.replace(MAG, learn_bias_min_rate_deg_s=None)
PITOT, WIND = aiding.PitotConfig(), aiding.WindModel(rw=0.1)
INIT_COURSE = fusion.InitConfig(yaw_source="gnss_course")
INIT_MAG = fusion.InitConfig(yaw_source="magnetometer", yaw_sigma_deg=15.0)

# key: (label, init, aiding)
CASES = {
    "E1": ("E1 : GNSS seul, cap initial = route GNSS", INIT_COURSE, None),
    "E2": ("E2 : + magnétomètre", INIT_MAG, aiding.AidingConfig(mag=MAG)),
    "E2b": ("E2b : magnétomètre, biais appris en permanence", INIT_MAG, aiding.AidingConfig(mag=MAG_ALWAYS)),
    "E3": ("E3 : + baromètre", INIT_MAG, aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ)),
    "E4": ("E4 : + Pitot et vent", INIT_MAG,
           aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ, pitot=PITOT, wind=WIND)),
    "E5a": ("E5a : + dérapage nul, σ 2° (naïf)", INIT_MAG,
            aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ, pitot=PITOT, wind=WIND, sideslip_sigma_deg=2.0)),
    # The real sideslip reaches 1.5 deg in turns and stays there for seconds:
    # fused at 10 Hz as white noise, 2 deg claims ten times too much
    # information. 6 deg ~ 2 deg * sqrt(10), i.e. 2 deg once per second.
    "E5i": ("E5i : σ 6°, vent initial nul", INIT_MAG,
            aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ, pitot=PITOT,
                                wind=dataclasses.replace(WIND, init_from_airspeed=False), sideslip_sigma_deg=6.0)),
    "E5": ("E5 : + dérapage nul, σ 6°", INIT_MAG,
           aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ, pitot=PITOT, wind=WIND, sideslip_sigma_deg=6.0)),
}


def turn_intervals(tr, threshold_deg=2.0):
    turning = np.abs(np.degrees(tr.euler[:, 0])) > threshold_deg
    edges = np.flatnonzero(np.diff(np.concatenate([[0], turning.astype(int), [0]])))
    last = len(tr.t) - 1
    return [(tr.t[a], tr.t[min(b, last)]) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def shade(ax, turns, label=False):
    for i, (a, b) in enumerate(turns):
        ax.axvspan(a, b, color=TURN_SHADE, lw=0, zorder=0, label="virages" if (label and i == 0) else None)


def rms_over_runs(x):
    return np.sqrt(np.mean(x**2, axis=0))


def along_cross(tr, lg, key="wind"):
    """Wind error projected on the air-relative track (along) and across it."""
    sl = lg.blocks[key]
    k = np.searchsorted(tr.t, lg.t)
    v_air = tr.v_n[k, :2] - tr.wind_n[k, :2]
    u = v_air / np.linalg.norm(v_air, axis=1, keepdims=True)
    e = lg.err[:, :, sl]
    along = e[..., 0] * u[:, 0] + e[..., 1] * u[:, 1]
    cross = -e[..., 0] * u[:, 1] + e[..., 1] * u[:, 0]
    return along, cross


def fig_heading(tr, logs, turns, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    ax = axes[0]
    for key, c in (("E1", C2), ("E2", C1)):
        lg = logs[key]
        ax.plot(lg.t, rms_over_runs(lg.err[:, :, 8]) * DEG, color=c, label=CASES[key][0])
    ax.set_yscale("log")
    ax.set_ylabel("Erreur de cap, RMS [°]")
    ax.set_title("Cap initial dans 5 m/s de vent : route GNSS ou magnétomètre")
    ax.legend(loc="upper right")
    ax = axes[1]
    for key, c in (("E2b", C2), ("E2", C1)):
        lg = logs[key]
        ax.plot(lg.t, rms_over_runs(lg.err[:, :, 8]) * DEG, color=c, label=f"{key} : erreur réelle (RMS)")
        ax.plot(lg.t, lg.sigma[:, :, 8].mean(axis=0) * DEG, color=c, lw=1.1, ls="--",
                label=f"{key} : 1σ annoncé")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 4)
    ax.set_ylabel("Cap [°]")
    ax.set_xlabel("Temps [s]")
    ax.set_title("Avant le premier virage : biais appris en permanence (E2b) ou seulement en virage (E2)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    for i, ax in enumerate(axes):
        shade(ax, turns, label=(i == 0))
    axes[0].set_xlim(0, tr.t[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_wind(tr, logs, turns, path, r=0):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.4), sharex=True)
    lg = logs["E5"]
    sl = lg.blocks["wind"]
    k = np.searchsorted(tr.t, lg.t)
    for i, (name, c) in enumerate((("nord", C1), ("est", C2))):
        est = tr.wind_n[k, i] - lg.err[r, :, sl.start + i]
        sig = lg.sigma[r, :, sl.start + i]
        axes[0].fill_between(lg.t, est - 3 * sig, est + 3 * sig, color=c, alpha=0.12, lw=0)
        axes[0].plot(lg.t, est, color=c, label=f"vent {name} estimé (E5, un vol)")
        axes[0].plot(tr.t, tr.wind_n[:, i], color=c, lw=1.0, ls=(0, (3, 2)))
    axes[0].set_ylabel("Vent [m/s]")
    axes[0].set_ylim(-8, 9)
    axes[0].set_title("Vent estimé (vrai en pointillés, ±3σ en bande) ; rotation du vent entre 300 et 360 s")
    axes[0].legend(loc="upper right", ncol=2)
    for ax, comp, title in ((axes[1], 0, "Composante le long de la trajectoire air"),
                            (axes[2], 1, "Composante de travers")):
        for key, c in (("E4", C2), ("E5a", C3), ("E5", C1)):
            err = along_cross(tr, logs[key])[comp]
            ax.plot(logs[key].t, rms_over_runs(err), color=c, label=CASES[key][0])
        ax.set_ylabel("Erreur de vent, RMS [m/s]")
        ax.set_title(title)
        ax.set_ylim(0, 1.5)
        ax.legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("Temps [s]")
    for ax in axes:
        shade(ax, turns, label=False)
        ax.set_xlim(0, tr.t[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(tr, lg, first_turn):
    late = lg.t > 120.0
    early = (lg.t > 5.0) & (lg.t < first_turn)
    oe = lg.out_err[:, late]
    s = {
        "pos_h": float(np.sqrt((oe[..., 0:2] ** 2).sum(-1).mean())),
        "pos_v": float(np.sqrt((oe[..., 2] ** 2).mean())),
        "yaw_early": float(np.sqrt((lg.err[:, early, 8] ** 2).mean()) * DEG),
        "yaw": float(np.sqrt((oe[..., 8] ** 2).mean()) * DEG),
        "nees_att_early": float(lg.nees["attitude"][:, early].mean() / 3),
        "nees_att": float(lg.nees["attitude"][:, late].mean() / 3),
    }
    for name, dof in (("mag_bias", 3), ("baro_bias", 1), ("wind", 2)):
        if name in lg.nees:
            s[f"nees_{name}"] = float(lg.nees[name][:, late].mean() / dof)
    if "mag_bias" in lg.nees:
        s["nees_mag_bias_early"] = float(lg.nees["mag_bias"][:, early].mean() / 3)
    if "wind" in lg.blocks:
        along, cross = along_cross(tr, lg)
        s["wind_along"] = float(np.sqrt((along[:, late] ** 2).mean()))
        s["wind_cross"] = float(np.sqrt((cross[:, late] ** 2).mean()))
        s["wind_cross_early"] = float(np.sqrt((cross[:, early] ** 2).mean()))
    for name, (t, v) in lg.aux_nis.items():
        dof = {"baro": 1, "mag": 3, "pitot": 1, "sideslip": 1}[name]
        s[f"nis_{name}"] = float(v[:, t > 120.0].mean() / dof)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    tr = trajectory3d.generate(trajectory3d.MISSION_WIND)
    turns = turn_intervals(tr)
    first_turn = turns[0][0]
    imu_cfg = imu.IMUConfig()
    logs, stats = {}, {}
    for key, (_label, init, ad) in CASES.items():
        logs[key] = fusion.run(tr, imu_cfg, gnss.REALISTIC, init, args.runs, np.random.default_rng(args.seed),
                               model_gnss_bias=True, latency_mode="delayed", aiding=ad)
        stats[key] = summary(tr, logs[key], first_turn)
        print(key, {k: round(v, 3) for k, v in stats[key].items()}, flush=True)

    fig_heading(tr, logs, turns, img / "step5_heading.png")
    fig_wind(tr, logs, turns, img / "step5_wind.png")

    def f(key, name, fmt="{:.2f}"):
        v = stats[key].get(name)
        return "–" if v is None else fmt.format(v)

    crab = np.degrees(np.arctan2(tr.v_n[0, 1], tr.v_n[0, 0]) - tr.euler[0, 2])
    lines = [
        "# Étape 5 : résultats",
        "",
        f"Généré par `scripts/step5_aiding.py`. Mission de {tr.t[-1]:.0f} s dans un vent de 5 m/s d'ouest "
        f"qui tourne au nord-ouest (6 m/s) entre 300 et 360 s ; écart initial entre route et cap : "
        f"{crab:.1f}°. GNSS réaliste, biais GNSS estimé, horizon retardé. {args.runs} runs par configuration, "
        f"mêmes tirages IMU et GNSS pour toutes. Premier virage à {first_turn:.0f} s.",
        "",
        "## Cap, position, cohérence",
        "",
        "| Configuration | Cap avant 1er virage [°] | Cap après 120 s [°] | NEES att. avant | NEES att. après "
        "| Position horiz. [m] | Position vert. [m] |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in CASES:
        lines.append(f"| {CASES[key][0]} | {f(key, 'yaw_early')} | {f(key, 'yaw')} | {f(key, 'nees_att_early')} "
                     f"| {f(key, 'nees_att')} | {f(key, 'pos_h')} | {f(key, 'pos_v')} |")
    lines += [
        "",
        "## États ajoutés (après 120 s, sauf mention)",
        "",
        "| Configuration | NEES biais magnéto avant 1er virage | NEES biais magnéto | NEES dérive baro | NEES vent | Vent le long [m/s] "
        "| Vent de travers [m/s] | Vent de travers avant 1er virage [m/s] | NIS magnéto | NIS baro | NIS Pitot |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("E2", "E2b", "E3", "E4", "E5a", "E5i", "E5"):
        lines.append(f"| {CASES[key][0]} | {f(key, 'nees_mag_bias_early')} | {f(key, 'nees_mag_bias')} | {f(key, 'nees_baro_bias')} "
                     f"| {f(key, 'nees_wind')} | {f(key, 'wind_along')} | {f(key, 'wind_cross')} "
                     f"| {f(key, 'wind_cross_early')} "
                     f"| {f(key, 'nis_mag')} | {f(key, 'nis_baro')} | {f(key, 'nis_pitot')} |")
    lines.append("")
    prov = provenance.build(__file__, params={"mission": trajectory3d.MISSION_WIND, "imu": imu_cfg,
                                              "gnss": gnss.REALISTIC,
                                              "cases": {k: {"init": v[1], "aiding": v[2]} for k, v in CASES.items()}},
                            seed=args.seed, runs=args.runs)
    provenance.write(prov, ROOT / "docs" / "step5_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step5_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
