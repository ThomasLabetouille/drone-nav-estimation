"""Step 7: replay of real PX4 flight logs, comparison with the onboard EKF2.

The logs are public, on review.px4.io. They are not in the repository; put
them in a folder (default: logs_px4/ next to the repository):

    https://review.px4.io/download?log=7ce4abb5-cacd-4732-a302-acd3a265deed
    https://review.px4.io/download?log=178e9452-5ce8-4024-9ef5-1e9ce6da4cd9
    https://review.px4.io/download?log=e163c6fb-3ae1-4355-a833-b2a9b87a6a26

    pip install -e ".[replay]"
    python scripts/step7_replay.py                   # ~20 min
    python scripts/step7_replay.py --logs D:/logs    # other folder
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navsim import provenance, replay, ulog_reader
from navsim.aiding import heading_from_magnetometer
from navsim.rotations import euler_from_quat, quat_conj, quat_rotate

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
    "legend.fontsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.5,
    "font.family": "DejaVu Sans",
})

# review.px4.io id -> (short name, sha256 of the file used for the results)
LOGS = {
    "7ce4abb5-cacd-4732-a302-acd3a265deed": ("A1", "fe9a21a577d90b7edf4a79bd99530f26f46f6ff82491d2615eec01fbe8e898fe"),
    "178e9452-5ce8-4024-9ef5-1e9ce6da4cd9": ("A2", "dbba98ef326591623cf70b60eaeb7c289b4b8bc80221e30df9686fda97a9dc2b"),
    "e163c6fb-3ae1-4355-a833-b2a9b87a6a26": ("B", "f13e448bb7da51fe9dd66821ad83a7d4a693b94b8b563907181ef654f9aaa51c"),
}
MAIN = ("A1", "A2")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def in_flight(t, data, margin=20.0):
    a, b = data.info["airborne"]
    return (t > a + margin) & (t < b - margin)


def agreement(log, data):
    c = replay.compare(log, data)
    m = in_flight(c["t"], data)
    out = {
        "horiz": rms(np.linalg.norm(c["dp"][m, :2], axis=1)),
        "vert": rms(c["dp"][m, 2]),
        "vel": rms(np.linalg.norm(c["dv"][m], axis=1)),
        "tilt": rms(np.linalg.norm(c["datt"][m, :2], axis=1)) * DEG,
        "head": rms(c["datt"][m, 2]) * DEG,
        # compare() gives reference minus estimate; shown as estimate minus EKF2
        "head_mean": -float(np.mean(c["datt"][m, 2])) * DEG,
    }
    if "dwind" in c:
        out["wind"] = rms(np.linalg.norm(c["dwind"][m], axis=1))
    return out, c


def nis_table(log, data):
    dof = {"gnss": 6, "mag": 3, "mag_heading": 1, "baro": 1, "pitot": 1, "sideslip": 1}
    out = {}
    for name, (t, v, r) in log.nis.items():
        m = in_flight(t, data)
        out[name] = (float(np.mean(v[m]) / dof[name]), 100 * float(np.mean(r[m])))
    return out


def data_findings(data):
    """What the log says about its own sensors, using EKF2's attitude, wind
    and velocity as the reference (no replay involved)."""
    k = np.clip(np.searchsorted(data.t_ref, data.t_mag), 0, len(data.t_ref) - 1)
    m_ned = quat_rotate(data.ref_q[k], data.mag)
    air = in_flight(data.t_mag, data)
    mean = m_ned[air].mean(axis=0)
    roll, pitch, yaw = euler_from_quat(data.ref_q[k])
    f = data.field_ned
    head_err = np.angle(np.exp(1j * (heading_from_magnetometer(data.mag, roll, pitch, np.arctan2(f[1], f[0]))
                                     - yaw))) * DEG
    v_air = data.ref_v - np.column_stack([data.ref_wind, np.zeros(len(data.t_ref))])
    v_b = quat_rotate(quat_conj(data.ref_q), v_air)
    V = np.linalg.norm(v_b, axis=1)
    beta = np.arcsin(v_b[:, 1] / np.maximum(V, 1e-3)) * DEG
    r_ref = np.array(euler_from_quat(data.ref_q)[0]) * DEG
    lvl = in_flight(data.t_ref, data) & (V > 10) & (np.abs(r_ref) < 5)
    kt = np.clip(np.searchsorted(data.t_ref, data.t_tas), 0, len(data.t_ref) - 1)
    ft = in_flight(data.t_tas, data) & (data.tas > 10)
    return {
        "mag_norm": float(np.linalg.norm(m_ned[air], axis=1).mean()),
        "wmm_norm": float(np.linalg.norm(f)),
        "mag_incl": float(np.degrees(np.arctan2(mean[2], np.hypot(mean[0], mean[1])))),
        "wmm_incl": float(np.degrees(np.arctan2(f[2], np.hypot(f[0], f[1])))),
        "mag_head_std": float(np.std(head_err[air])),
        "mag_head_mean": float(np.mean(head_err[air])),
        "beta_level": float(np.mean(beta[lvl])),
        "tas_ratio": float(np.median(data.tas[ft] / V[kt][ft])),
        "t_head": data.t_mag, "head_err": head_err,
    }


# --- figures ------------------------------------------------------------------

def fig_outages(runs, path):
    fig, axes = plt.subplots(len(MAIN), 1, figsize=(10, 3.6 * len(MAIN)))
    for ax, name in zip(np.atleast_1d(axes), MAIN, strict=True):
        r = runs[name]
        for key, col, lab in (("R1", C2, "IMU + GNSS"), ("R2", C4, "+ baro, magnétomètre"),
                              ("R3", C1, "+ Pitot, vent, dérapage")):
            lg = r["out_" + key]
            h = np.linalg.norm(lg.p[:, :2] - lg.gnss[:, :2], axis=1)
            dead = lg.gnss_status == 2
            ax.plot(lg.t, np.where(dead, h, np.nan), color=col, label=lab)
        for a, b in r["windows"]:
            ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 300)
        ax.set_ylabel("Écart au GNSS [m]")
        ax.set_title(f"Vol {name} : pertes GNSS simulées de 30 s, écart entre l'estimation et le vrai GNSS")
        ax.legend(loc="upper left", ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("Temps depuis le début du log [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sensors(runs, path):
    name = MAIN[0]
    r = runs[name]
    data, fnd = r["data"], r["findings"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.6), sharex=True)
    a, b = data.info["airborne"]
    ax = axes[0]
    air = (fnd["t_head"] > a) & (fnd["t_head"] < b)
    ax.plot(fnd["t_head"][air], fnd["head_err"][air], color=C4, lw=1.0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("[°]")
    ax.set_title("Cap du magnétomètre (compensé en inclinaison) moins cap de l'EKF2, en vol")
    ax = axes[1]
    for key, col, lab in (("full_nooffset", C2, "dérapage nul imposé (σ 6°)"), ("R3", C1, "avec état de décalage")):
        c = r["cmp_" + key]
        m = in_flight(c["t"], data, 0)
        ax.plot(c["t"][m], -c["datt"][m, 2] * DEG, color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("[°]")
    ax.set_title("Cap estimé moins cap de l'EKF2")
    ax.legend(loc="upper left")
    ax = axes[2]
    lg = r["R3"]
    sl = lg.blocks["beta_offset"]
    b_est = lg.extra[:, sl.start - 15] * DEG
    b_sig = lg.sigma[:, sl.start] * DEG
    m = lg.t > (lg.wind_started or 0)
    ax.fill_between(lg.t[m], (b_est - 3 * b_sig)[m], (b_est + 3 * b_sig)[m], color=C1, alpha=0.15, lw=0)
    ax.plot(lg.t[m], b_est[m], color=C1, label="décalage estimé (±3σ)")
    ax.axhline(fnd["beta_level"], color=C2, ls="--", lw=1.2, label="dérapage déduit des états de l'EKF2, ailes à plat")
    ax.set_ylabel("[°]")
    ax.set_xlabel("Temps depuis le début du log [s]")
    ax.set_title("Décalage de dérapage estimé")
    ax.legend(loc="lower right")
    for ax in axes:
        ax.set_xlim(a - 10, b + 10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=ROOT.parent / "logs_px4")
    args = ap.parse_args()
    img = ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)

    paths, hashes = {}, {}
    for lid, (name, expected) in LOGS.items():
        p = args.logs / f"{lid}.ulg"
        if not p.exists():
            sys.exit(f"{p} manquant : télécharger https://review.px4.io/download?log={lid}")
        hashes[name] = sha256(p)
        if hashes[name] != expected:
            print(f"attention : {p.name} n'est pas le fichier des résultats publiés ({hashes[name][:12]})")
        paths[name] = p

    runs = {}
    for name in MAIN:
        data = ulog_reader.load(paths[name], name)
        r = {"data": data, "findings": data_findings(data)}
        cfgs = {
            "R1": replay.config_like_ekf2(data, air_data=False, mag=None),
            "R2": replay.config_like_ekf2(data, air_data=False, mag="heading"),
            "R3": replay.config_like_ekf2(data),
            "full_nooffset": replay.config_like_ekf2(data, sideslip_offset=False),
            "full_beta17": replay.config_like_ekf2(data, sideslip_sigma_deg=17.0, sideslip_offset=False),
            "full_nobeta": replay.config_like_ekf2(data, sideslip_sigma_deg=None),
            "full_mag3d": replay.config_like_ekf2(data, mag="3d"),
        }
        for key, cfg in cfgs.items():
            lg = replay.run(data, cfg)
            r[key] = lg
            r["agr_" + key], r["cmp_" + key] = agreement(lg, data)
            r["nis_" + key] = nis_table(lg, data)
            print(name, key, {k: round(v, 3) for k, v in r["agr_" + key].items()}, flush=True)
        r["windows"] = replay.outage_windows(data.info["airborne"])
        cfgs["R1b"] = dataclasses.replace(cfgs["R1"], baro=cfgs["R2"].baro)
        for key in ("R1", "R1b", "R2", "R3"):
            lg = replay.run(data, dataclasses.replace(cfgs[key], gnss_outage=r["windows"]))
            r["out_" + key] = lg
            r["oerr_" + key] = replay.outage_errors(lg, r["windows"])
            print(name, "outage", key, r["oerr_" + key].round(1).tolist(), flush=True)
        runs[name] = r

    # log B: airspeed scale
    dB = ulog_reader.load(paths["B"], "B")
    fB = data_findings(dB)
    lgB = replay.run(dB, replay.config_like_ekf2(dB))
    sB = float(lgB.extra[-1, lgB.blocks["tas_scale"].start - 15])
    sB_sig = float(lgB.sigma[-1, lgB.blocks["tas_scale"].start])
    print("B", fB["tas_ratio"], sB, sB_sig)

    fig_outages(runs, img / "step7_outages.png")
    fig_sensors(runs, img / "step7_sensors.png")

    def f(x, fmt="{:.2f}"):
        return "–" if x is None else fmt.format(x)

    lines = [
        "# Étape 7 : résultats",
        "",
        "Généré par `scripts/step7_replay.py` sur des logs publics de review.px4.io. Écarts calculés en vol "
        "(du décollage + 20 s à l'atterrissage − 20 s).",
        "",
        "## Logs",
        "",
        "| Vol | Log | Matériel | PX4 | Durée [s] | En vol [s] | IMU [Hz] | GNSS loggé [Hz] | Magnéto loggé [Hz] "
        "| Vitesse air loggée [Hz] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lid, (name, _) in LOGS.items():
        d = runs[name]["data"] if name in runs else dB
        i = d.info
        a, b = i["airborne"]
        lines.append(f"| {name} | `{lid[:8]}` | {i['sys'].get('ver_hw', '?')} | `{i['sys'].get('ver_sw', '?')[:7]}` "
                     f"| {i['duration_s']:.0f} | {b - a:.0f} | {i['imu_rate_hz']:.0f} | {i['gnss_rate_hz']:.1f} "
                     f"| {i['mag_rate_hz']:.1f} | {i['airspeed_rate_hz']:.1f} |")
    lines += [
        "",
        "## Ce que les logs disent de leurs capteurs (référence : états de l'EKF2)",
        "",
        "| Vol | Champ mesuré [G] | Champ WMM [G] | Inclinaison mesurée [°] | Inclinaison WMM [°] "
        "| Cap magnéto − cap EKF2, écart-type [°] | Dérapage ailes à plat [°] | Vitesse air / |v − w| |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, fnd in [(n, runs[n]["findings"]) for n in MAIN] + [("B", fB)]:
        lines.append(f"| {name} | {fnd['mag_norm']:.3f} | {fnd['wmm_norm']:.3f} | {fnd['mag_incl']:.1f} "
                     f"| {fnd['wmm_incl']:.1f} | {fnd['mag_head_std']:.1f} | {fnd['beta_level']:.1f} "
                     f"| {fnd['tas_ratio']:.3f} |")
    lines += [
        "",
        "## Écart au filtre embarqué (RMS en vol)",
        "",
        "| Vol | Configuration | Position horiz. [m] | Position vert. [m] | Vitesse [m/s] | Inclinaison [°] "
        "| Cap [°] | Cap, moyenne [°] | Vent [m/s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {"R1": "R1 : IMU + GNSS", "R1b": "R1 + baro", "R2": "R2 : + baro, magnétomètre (cap)",
              "R3": "R3 : + Pitot, vent, dérapage avec décalage",
              "full_nooffset": "R3 sans état de décalage (dérapage σ 6°)",
              "full_beta17": "R3 sans décalage, dérapage σ 17° (valeur EKF2)",
              "full_nobeta": "R3 sans dérapage",
              "full_mag3d": "R3, magnétomètre 3 axes contre le WMM"}
    for name in MAIN:
        for key, lab in labels.items():
            if "agr_" + key not in runs[name]:
                continue
            a = runs[name]["agr_" + key]
            lines.append(f"| {name} | {lab} | {a['horiz']:.2f} | {a['vert']:.2f} | {a['vel']:.2f} | {a['tilt']:.2f} "
                         f"| {a['head']:.2f} | {a['head_mean']:.2f} | {f(a.get('wind'))} |")
    lines += [
        "",
        "## Cohérence sur données réelles (NIS moyen par degré de liberté, en vol ; rejets au test à 99,9 %)",
        "",
        "| Vol | Configuration | GNSS | Baro | Magnéto (cap) | Magnéto (3 axes) | Pitot | Dérapage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in MAIN:
        for key in ("R3", "full_nooffset", "full_mag3d"):
            nis = runs[name]["nis_" + key]
            cell = lambda s, nis=nis: "–" if s not in nis else f"{nis[s][0]:.2f} ({nis[s][1]:.1f} %)"
            lines.append(f"| {name} | {labels[key]} | {cell('gnss')} | {cell('baro')} | {cell('mag_heading')} "
                         f"| {cell('mag')} | {cell('pitot')} | {cell('sideslip')} |")
    lines += [
        "",
        "## Calibrations estimées en vol",
        "",
        "| Vol | Décalage de dérapage estimé [°] | ±1σ [°] | Dérapage déduit de l'EKF2 [°] | Échelle de vitesse air "
        "estimée | Rapport vitesse air / |v − w| de l'EKF2 |",
        "|---|---|---|---|---|---|",
    ]
    for name in MAIN:
        lg, fnd = runs[name]["R3"], runs[name]["findings"]
        sl, ss = lg.blocks["beta_offset"], lg.blocks["tas_scale"]
        lines.append(f"| {name} | {lg.extra[-1, sl.start - 15] * DEG:.1f} | {lg.sigma[-1, sl.start] * DEG:.1f} "
                     f"| {fnd['beta_level']:.1f} | {1 + lg.extra[-1, ss.start - 15]:.3f} | {fnd['tas_ratio']:.3f} |")
    lines.append(f"| B | – | – | – | {1 + sB:.3f} ± {sB_sig:.3f} | {fB['tas_ratio']:.3f} |")
    lines += [
        "",
        "## Pertes GNSS simulées sur les vols réels (30 s, toutes les 90 s en vol)",
        "",
        "Écart horizontal entre l'estimation à la fin de la perte et la première position GNSS qui suit.",
        "",
        "| Configuration | Pertes | Écart médian [m] | Médiane A1 / A2 [m] | Écart max [m] | 1σ annoncé médian [m] "
        "| Écart / σ max |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("R1", "R1b", "R2", "R3"):
        e = np.concatenate([runs[n]["oerr_" + key] for n in MAIN])
        per = " / ".join(f"{np.median(runs[n]['oerr_' + key][:, 0]):.1f}" for n in MAIN)
        lines.append(f"| {labels[key]} | {len(e)} | {np.median(e[:, 0]):.1f} | {per} | {e[:, 0].max():.1f} "
                     f"| {np.median(e[:, 1]):.1f} | {np.max(e[:, 0] / e[:, 1]):.2f} |")
    lines.append("")

    prov = provenance.build(__file__, params={
        "logs": {name: {"review_px4_id": lid, "sha256": hashes[name]} for lid, (name, _) in LOGS.items()},
        "configs": {n: {k: replay.config_like_ekf2(runs[n]["data"]) for k in ("R3",)} for n in MAIN},
    }, seed=0, runs=1)
    provenance.write(prov, ROOT / "docs" / "step7_provenance.json")
    lines += provenance.markdown(prov)
    (ROOT / "docs" / "step7_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
