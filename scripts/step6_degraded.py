"""Step 6: degraded modes. GNSS outage, GNSS jump, magnetic disturbance,
wind change; innovation gating and its failure modes.

All runs fly the step 5 mission in wind, with the same IMU, GNSS and aiding
draws. Unless stated, the filter has every step 5 sensor and gates every
measurement at 99.9 %.

    python scripts/step6_degraded.py              # 40 runs per configuration, ~1 h
    python scripts/step6_degraded.py --runs 10    # quicker
    python scripts/step6_degraded.py --no-video   # skip the animation
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim import aiding, fusion, gnss, imu, provenance, trajectory3d
from navsim.faults import Faults

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
SHADE = "#f3d9cf"
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

GATE = 0.999
OUTAGE = (180.0, 240.0)
JUMP = (320.0, 350.0, (0.0, 15.0, 0.0))
MAG_STRONG = (170.0, 230.0, (0.05, 0.03, 0.0))      # 0.058 G, 12 % of the field
MAG_WEAK = (170.0, 230.0, (0.008, 0.005, 0.0))      # 0.009 G, 2 % of the field

MAG, PITOT = aiding.MagConfig(), aiding.PitotConfig()
INIT_COURSE = fusion.InitConfig(yaw_source="gnss_course")
INIT_MAG = fusion.InitConfig(yaw_source="magnetometer", yaw_sigma_deg=15.0)


def full(**wind):
    return aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ, pitot=PITOT, wind=aiding.WindModel(**wind),
                               sideslip_sigma_deg=6.0)


STEP5 = full(rw=0.1)                                # step 5 E5
FROZEN = full(rw=0.1, rw_without_gnss=0.01)
O_FAULT = Faults(gnss_outage=(OUTAGE,))

RESET = 45.0   # [s] GNSS lock-out protection, the default of fusion.run
J_FAULT = Faults(gnss_jump=(JUMP,))

# key: (label, init, aiding, faults, gate, GNSS reset delay)
CASES = {
    "O1": ("O1 : IMU + GNSS", INIT_COURSE, None, O_FAULT, GATE, RESET),
    "O2": ("O2 : + magnétomètre et baro", INIT_MAG, aiding.AidingConfig(mag=MAG, baro=aiding.BARO_10HZ),
           O_FAULT, GATE, RESET),
    "O3": ("O3 : + Pitot et vent (étape 5)", INIT_MAG, STEP5, O_FAULT, GATE, RESET),
    "O4": ("O4 : + vent quasi figé sans GNSS", INIT_MAG, FROZEN, O_FAULT, GATE, RESET),
    "W1": ("W1 : marche aléatoire 0,01, sans protection", INIT_MAG,
           full(rw=0.01, reset_after_rejected_s=None), Faults(), GATE, RESET),
    "W2": ("W2 : marche aléatoire 0,01, avec protection", INIT_MAG, full(rw=0.01), Faults(), GATE, RESET),
    "J1": ("J1 : sans test d'innovation", INIT_MAG, STEP5, J_FAULT, None, RESET),
    "J2": ("J2 : avec test, vent de l'étape 5", INIT_MAG, STEP5, J_FAULT, GATE, RESET),
    "J3": ("J3 : avec test, vent quasi figé sans GNSS, sans réinitialisation", INIT_MAG, FROZEN, J_FAULT, GATE,
           None),
    "J4": ("J4 : comme J3, réinitialisation sur le GNSS après 45 s", INIT_MAG, FROZEN, J_FAULT, GATE, RESET),
    "J5": ("J5 : comme J2, réinitialisation après 10 s", INIT_MAG, STEP5, J_FAULT, GATE, 10.0),
    "M1": ("M1 : perturbation de 0,058 G, sans test", INIT_MAG, STEP5, Faults(mag_disturbance=(MAG_STRONG,)),
           None, RESET),
    "M2": ("M2 : perturbation de 0,058 G, avec test", INIT_MAG, STEP5, Faults(mag_disturbance=(MAG_STRONG,)),
           GATE, RESET),
    "M3": ("M3 : perturbation de 0,009 G, avec test", INIT_MAG, STEP5, Faults(mag_disturbance=(MAG_WEAK,)),
           GATE, RESET),
}


def at(lg, t):
    return int(np.searchsorted(lg.t, t - 1e-9))


def horiz(lg, k):
    """Horizontal error of the delivered state, its RMS over runs, and the
    announced 1 sigma (from the filter covariance)."""
    h = np.linalg.norm(lg.out_err[:, k, :2], axis=1)
    sig = np.sqrt(lg.sigma[:, k, 0] ** 2 + lg.sigma[:, k, 1] ** 2)
    return h, sig


def window(lg, a, b):
    return (lg.t >= a) & (lg.t < b)


def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


# --- figures ------------------------------------------------------------------

def shade(ax, a, b, label=None):
    ax.axvspan(a, b, color=SHADE, lw=0, zorder=0, label=label)


def fig_outage(logs, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for key, c in (("O1", C2), ("O2", C4), ("O3", C1), ("O4", C3)):
        lg = logs[key]
        h = np.linalg.norm(lg.out_err[..., :2], axis=2)
        sig = np.sqrt(lg.sigma[..., 0] ** 2 + lg.sigma[..., 1] ** 2).mean(axis=0)
        ax.plot(lg.t_out, rms(h, 0), color=c, label=CASES[key][0])
        ax.plot(lg.t, sig, color=c, lw=1.0, ls="--")
    shade(ax, *OUTAGE, label="GNSS perdu")
    ax.set_yscale("log")
    ax.set_xlim(150, 300)
    ax.set_ylim(0.5, 200)
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Erreur horizontale [m]")
    ax.set_title("60 s sans GNSS : erreur réelle (RMS, trait plein) et 1σ annoncé (tirets)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_wind(logs, tr, path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for key, c in (("O3", C1), ("W1", C2), ("W2", C3)):
        lg = logs[key]
        e = np.linalg.norm(lg.err[..., lg.blocks["wind"]], axis=2)
        ax.plot(lg.t, rms(e, 0), color=c, label=CASES[key][0] if key != "O3" else "marche aléatoire 0,1 (étape 5)")
    shade(ax, 300, 360, label="rotation du vent")
    ax.set_yscale("log")
    ax.set_xlim(250, tr.t[-1])
    ax.set_ylim(0.01, 10)
    ax.set_xlabel("Temps [s]")
    ax.set_ylabel("Erreur de vent, RMS [m/s]")
    ax.set_title("Marche aléatoire du vent trop faible : le test d'innovation bloque le Pitot")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gating(logs, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    ax = axes[0]
    for key, c in (("J1", C2), ("J2", C1), ("J3", "#c0392b"), ("J4", C4), ("J5", C3)):
        lg = logs[key]
        h = np.linalg.norm(lg.out_err[..., :2], axis=2)
        ax.plot(lg.t_out, rms(h, 0), color=c, label=CASES[key][0])
    shade(ax, JUMP[0], JUMP[1], label="saut de 15 m")
    ax.axvspan(300, 360, ymin=0, ymax=0.04, color=INK2, lw=0, label="rotation du vent")
    ax.set_yscale("log")
    ax.set_xlim(300, 480)
    ax.set_ylim(0.3, 1000)
    ax.set_ylabel("Erreur horizontale, RMS [m]")
    ax.set_title("GNSS décalé de 15 m pendant 30 s, pendant que le vent tourne")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    for ax, keys, title in (
            (axes[1], (("M1", C2), ("M2", C1)), "Perturbation magnétique forte (0,058 G pendant 60 s)"),
            (axes[2], (("M3", C4),), "Perturbation faible (0,009 G) : sous le seuil du test")):
        for key, c in keys:
            lg = logs[key]
            ax.plot(lg.t, rms(lg.err[..., 8], 0) * DEG, color=c, label=CASES[key][0])
            ax.plot(lg.t, lg.sigma[..., 8].mean(axis=0) * DEG, color=c, lw=1.0, ls="--")
        shade(ax, MAG_STRONG[0], MAG_STRONG[1], label="perturbation")
        ax.set_xlim(150, 260)
        ax.set_ylabel("Erreur de cap [°]")
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[1].set_yscale("log")
    axes[2].set_ylim(0, 2)
    axes[2].set_xlabel("Temps [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- animation ----------------------------------------------------------------

def make_video(tr, logs, out_dir, r):
    """One flight through the GNSS outage, seen from a camera that follows
    the drone: where it is, and where two filters think it is."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg absent : pas de vidéo")
        return
    frames = out_dir / "_frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    keys = (("O1", C2, "IMU + GNSS"), ("O3", C1, "IMU + GNSS + magnétomètre, baro, Pitot"))
    lg0 = logs["O3"]
    k_true = np.searchsorted(tr.t, lg0.t_out)
    p_true = tr.p_n[k_true, :2]
    est = {key: p_true - logs[key].out_err[r, :, :2] for key, _, _ in keys}
    sig = {key: logs[key].sigma[r, :, :2] for key, _, _ in keys}
    t0, t1 = 165.0, 262.0
    idx = np.flatnonzero((lg0.t_out >= t0) & (lg0.t_out <= t1))
    trail = 40 * 5                       # 40 s of track behind the drone (5 Hz)
    half = 110.0                         # [m] half width of the chase view
    lost_seg = (tr.t >= OUTAGE[0]) & (tr.t < OUTAGE[1])
    still = int(np.searchsorted(lg0.t_out[idx], OUTAGE[1] - 0.4))

    plt.rcParams.update({"axes.titlesize": 13, "font.size": 11})
    for i, k in enumerate(idx):
        t = lg0.t_out[k]
        lost = OUTAGE[0] <= t < OUTAGE[1]
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=(1.1, 1), height_ratios=(1, 1.15), left=0.05, right=0.97,
                              top=0.82, bottom=0.08, wspace=0.16, hspace=0.38)
        fig.text(0.05, 0.94, "Perte du GNSS en vol : où le drone croit-il être ?", fontsize=18, weight="bold",
                 color=INK)
        status = (f"GNSS PERDU depuis {t - OUTAGE[0]:4.1f} s" if lost
                  else ("GNSS disponible" if t < OUTAGE[0] else "GNSS revenu"))
        fig.text(0.05, 0.895, f"t = {t:5.1f} s    {status}", fontsize=14,
                 color=("#c0392b" if lost else INK2), weight=("bold" if lost else "normal"))
        fig.text(0.97, 0.94, "voilure fixe, 17 à 22 m/s, vent de 5 m/s", fontsize=11, color=INK2, ha="right")
        fig.text(0.97, 0.905, "simulation, IMU MEMS, filtre de Kalman étendu", fontsize=11, color=INK2,
                 ha="right")

        # chase view: east/north relative to the current true position
        ax = fig.add_subplot(gs[:, 0])
        c = p_true[k]
        a = max(idx[0], k - trail)
        ax.plot(p_true[a:k + 1, 1] - c[1], p_true[a:k + 1, 0] - c[0], color=INK, lw=2.4, label="drone (vérité)")
        for key, col, lab in keys:
            e = est[key]
            ax.plot(e[a:k + 1, 1] - c[1], e[a:k + 1, 0] - c[0], color=col, lw=2.0, label=f"estimé : {lab}")
            s3 = 3 * sig[key][k]
            ax.add_patch(Ellipse((e[k, 1] - c[1], e[k, 0] - c[0]), 2 * s3[1], 2 * s3[0], fc=col, alpha=0.13,
                                 ec=col, lw=1.0))
            ax.plot(e[k, 1] - c[1], e[k, 0] - c[0], "o", color=col, ms=8)
        ax.plot(0, 0, marker=(3, 0, -np.degrees(np.arctan2(*(p_true[k] - p_true[max(k - 5, 0)])[::-1]))),
                color=INK, ms=13)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel("Est [m]")
        ax.set_ylabel("Nord [m]")
        ax.set_title("Caméra qui suit le drone (ellipses : incertitude annoncée, 3σ)")
        ax.legend(loc="lower left", fontsize=10)

        # whole mission, for context
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(tr.p_n[:, 1], tr.p_n[:, 0], color=GRID, lw=1.5)
        ax.plot(tr.p_n[lost_seg, 1], tr.p_n[lost_seg, 0], color="#c0392b", lw=2.5, label="segment sans GNSS")
        ax.plot(p_true[k, 1], p_true[k, 0], "o", color=INK, ms=7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Mission (8 min)")
        ax.legend(loc="lower right", fontsize=9)

        ax = fig.add_subplot(gs[1, 1])
        for j, (key, col, lab) in enumerate(keys):
            h = np.linalg.norm(est[key][idx[0]:k + 1] - p_true[idx[0]:k + 1], axis=1)
            ax.plot(lg0.t_out[idx[0]:k + 1], h, color=col, lw=2.0)
            ax.text(0.03, 0.93 - 0.14 * j, f"{h[-1]:5.1f} m   {lab}", transform=ax.transAxes, color=col,
                    fontsize=12, weight="bold", va="top")
        shade(ax, *OUTAGE)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("[m]")
        ax.set_title("Erreur de position horizontale")
        fig.savefig(frames / f"f{i:04d}.png")
        plt.close(fig)
    plt.rcParams.update({"axes.titlesize": 11, "font.size": 10})

    mp4 = out_dir / "step6_outage.mp4"
    gif = out_dir / "step6_outage.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frames / "f%04d.png"),
                    "-vf", "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];"
                           "[b][p]paletteuse=dither=bayer", str(gif)], check=True)
    # the frame just before GNSS comes back, for places where a video does not play
    shutil.copy(frames / f"f{still:04d}.png", out_dir / "step6_outage_still.png")
    shutil.rmtree(frames)


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    tr = trajectory3d.generate(trajectory3d.MISSION_WIND)
    imu_cfg = imu.IMUConfig()
    logs = {}
    for key, (_label, init, ad, faults, gate, reset) in CASES.items():
        logs[key] = fusion.run(tr, imu_cfg, gnss.REALISTIC, init, args.runs, np.random.default_rng(args.seed),
                               model_gnss_bias=True, latency_mode="delayed", aiding=ad, faults=faults,
                               gate_prob=gate, gnss_reset_after_s=reset)
        print(key, "ok", flush=True)

    fig_outage(logs, img / "step6_outage.png")
    fig_wind(logs, tr, img / "step6_wind.png")
    fig_gating(logs, img / "step6_gating.png")

    lines = [
        "# Étape 6 : résultats",
        "",
        f"Généré par `scripts/step6_degraded.py`. Mission de l'étape 5 (vent de 5 m/s), {args.runs} runs par "
        f"configuration, mêmes tirages pour toutes. Test d'innovation à {GATE * 100:.1f} % quand il est actif. "
        "Erreurs de l'état délivré en temps réel.",
        "",
        f"## Perte GNSS de {OUTAGE[0]:.0f} à {OUTAGE[1]:.0f} s",
        "",
        "| Configuration | Erreur horiz. après 30 s [m] | après 60 s [m] | pire run après 60 s [m] "
        "| 1σ annoncé après 60 s [m] | NEES position après 60 s | Mesures GNSS rejetées après retour |",
        "|---|---|---|---|---|---|---|",
    ]
    stats = {}
    for key in ("O1", "O2", "O3", "O4"):
        lg = logs[key]
        k30, k60 = at(lg, OUTAGE[0] + 30), at(lg, OUTAGE[1] - 0.2)
        h30, _ = horiz(lg, k30)
        h60, s60 = horiz(lg, k60)
        back = lg.gnss_status[:, window(lg, OUTAGE[1], OUTAGE[1] + 30)]
        stats[key] = dict(e60=rms(h60))
        lines.append(f"| {CASES[key][0]} | {rms(h30):.1f} | {rms(h60):.1f} | {h60.max():.1f} | {s60.mean():.1f} "
                     f"| {lg.nees['position'][:, at(lg, OUTAGE[1] - 0.2)].mean() / 3:.2f} "
                     f"| {100 * (back == 1).mean():.1f} % |")

    lines += [
        "",
        "## Vent pendant sa rotation (300 à 360 s), sans panne",
        "",
        "| Configuration | Erreur de vent max [m/s] | NEES vent 300-370 s | Erreur de vent après 400 s [m/s] "
        "| Pitot rejeté 300-400 s | Réinitialisations du vent |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("O3", "W1", "W2"):
        lg = logs[key]
        e = np.linalg.norm(lg.err[..., lg.blocks["wind"]], axis=2)
        w = window(lg, 300, 370)
        t_p = lg.aux_nis["pitot"][0]
        rej = lg.aux_rejected["pitot"][:, (t_p >= 300) & (t_p < 400)]
        label = CASES[key][0] if key != "O3" else "O3 : marche aléatoire 0,1 (étape 5)"
        lines.append(f"| {label} | {rms(e[:, w], 0).max():.2f} | {lg.nees['wind'][:, w].mean() / 2:.1f} "
                     f"| {rms(e[:, lg.t > 400]):.2f} | {100 * rej.mean():.1f} % "
                     f"| {sum(n for _, n in lg.wind_resets)} |")

    lines += [
        "",
        "## Saut GNSS de 15 m vers l'est, de 320 à 350 s",
        "",
        "| Configuration | Erreur horiz. max pendant le saut [m] | NEES position pendant le saut "
        "| Erreur horiz. max après le saut [m] | Mesures GNSS rejetées pendant le saut "
        "| Mesures GNSS rejetées après le saut | Réinitialisations sur le GNSS |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("J1", "J2", "J3", "J4", "J5"):
        lg = logs[key]
        w = window(lg, JUMP[0], JUMP[1])
        after = lg.t >= JUMP[1]
        h = np.linalg.norm(lg.out_err[..., :2], axis=2)
        lines.append(f"| {CASES[key][0]} | {rms(h[:, w], 0).max():.1f} | {lg.nees['position'][:, w].mean() / 3:.1f} "
                     f"| {rms(h[:, after], 0).max():.1f} | {100 * (lg.gnss_status[:, w] == 1).mean():.1f} % "
                     f"| {100 * (lg.gnss_status[:, after] == 1).mean():.1f} % "
                     f"| {sum(n for _, n in lg.gnss_resets)} |")

    lines += [
        "",
        "## Perturbation magnétique de 170 à 230 s",
        "",
        "| Configuration | Erreur de cap max [°] | NEES attitude pendant | Mesures magnéto rejetées pendant "
        "| Mesures magnéto rejetées hors panne |",
        "|---|---|---|---|---|",
    ]
    for key in ("M1", "M2", "M3"):
        lg = logs[key]
        w = window(lg, MAG_STRONG[0], MAG_STRONG[1] + 10)
        t_m = lg.aux_nis["mag"][0]
        rej = lg.aux_rejected["mag"]
        inside = (t_m >= MAG_STRONG[0]) & (t_m < MAG_STRONG[1])
        outside = ((t_m < MAG_STRONG[0]) | (t_m >= MAG_STRONG[1] + 30)) & (t_m > 5)
        lines.append(f"| {CASES[key][0]} | {rms(lg.err[:, w, 8], 0).max() * DEG:.2f} "
                     f"| {lg.nees['attitude'][:, w].mean() / 3:.1f} | {100 * rej[:, inside].mean():.1f} % "
                     f"| {100 * rej[:, outside].mean():.2f} % |")
    lines.append("")

    # The animated flight: the run whose O3 error at the end of the outage is the median one.
    lg = logs["O3"]
    h60, _ = horiz(lg, at(lg, OUTAGE[1] - 0.2))
    r = int(np.argsort(h60)[len(h60) // 2])
    lines += [f"Vol animé (`docs/img/step6_outage.mp4`) : run {r}, celui dont l'erreur de O3 à la fin de la "
              "perte est la médiane.", ""]
    if not args.no_video:
        make_video(tr, logs, img, r)

    prov = provenance.build(__file__, params={"mission": trajectory3d.MISSION_WIND, "imu": imu_cfg,
                                              "gnss": gnss.REALISTIC, "gate": GATE,
                                              "cases": {k: {"init": v[1], "aiding": v[2], "faults": v[3],
                                                            "gate": v[4], "gnss_reset_s": v[5]}
                                                        for k, v in CASES.items()}},
                            seed=args.seed, runs=args.runs)
    provenance.write(prov, ROOT / "docs" / "step6_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step6_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
