# Étape 7 : résultats

Généré par `scripts/step7_replay.py` sur des logs publics de review.px4.io. Écarts calculés en vol (du décollage + 20 s à l'atterrissage − 20 s).

## Logs

| Vol | Log | Matériel | PX4 | Durée [s] | En vol [s] | IMU [Hz] | GNSS loggé [Hz] | Magnéto loggé [Hz] | Vitesse air loggée [Hz] |
|---|---|---|---|---|---|---|---|---|---|
| A1 | `7ce4abb5` | PX4_FMU_V6X | `51277b3` | 705 | 321 | 200 | 2.0 | 2.0 | 5.0 |
| A2 | `178e9452` | PX4_FMU_V6X | `51277b3` | 636 | 417 | 200 | 2.0 | 2.0 | 5.0 |
| B | `e163c6fb` | PX4_FMU_V6C | `d6f12ad` | 593 | 583 | 203 | 7.5 | 2.0 | 5.0 |

## Ce que les logs disent de leurs capteurs (référence : états de l'EKF2)

| Vol | Champ mesuré [G] | Champ WMM [G] | Inclinaison mesurée [°] | Inclinaison WMM [°] | Cap magnéto − cap EKF2, écart-type [°] | Dérapage ailes à plat [°] | Vitesse air / |v − w| |
|---|---|---|---|---|---|---|---|
| A1 | 0.401 | 0.485 | 58.8 | 64.6 | 11.4 | -9.6 | 0.983 |
| A2 | 0.399 | 0.485 | 58.7 | 64.6 | 9.0 | -7.1 | 1.000 |
| B | 0.440 | 0.448 | 35.9 | 34.2 | 3.9 | 5.0 | 1.364 |

## Écart au filtre embarqué (RMS en vol)

| Vol | Configuration | Position horiz. [m] | Position vert. [m] | Vitesse [m/s] | Inclinaison [°] | Cap [°] | Cap, moyenne [°] | Vent [m/s] |
|---|---|---|---|---|---|---|---|---|
| A1 | R1 : IMU + GNSS | 0.45 | 0.14 | 0.11 | 0.76 | 0.68 | 0.17 | – |
| A1 | R2 : + baro, magnétomètre (cap) | 0.46 | 0.14 | 0.11 | 0.76 | 0.75 | 0.13 | – |
| A1 | R3 : + Pitot, vent, dérapage avec décalage | 0.44 | 0.15 | 0.12 | 0.76 | 0.87 | -0.06 | 0.55 |
| A1 | R3 sans état de décalage (dérapage σ 6°) | 0.48 | 0.18 | 0.21 | 0.93 | 3.28 | -3.02 | 1.32 |
| A1 | R3 sans décalage, dérapage σ 17° (valeur EKF2) | 0.45 | 0.15 | 0.12 | 0.76 | 1.24 | -0.91 | 0.71 |
| A1 | R3 sans dérapage | 0.44 | 0.14 | 0.11 | 0.76 | 0.75 | 0.15 | 0.55 |
| A1 | R3, magnétomètre 3 axes contre le WMM | 0.44 | 0.15 | 0.11 | 0.73 | 0.78 | -0.32 | 0.54 |
| A2 | R1 : IMU + GNSS | 0.19 | 0.12 | 0.09 | 0.94 | 0.96 | 0.42 | – |
| A2 | R2 : + baro, magnétomètre (cap) | 0.19 | 0.10 | 0.09 | 0.94 | 0.95 | 0.40 | – |
| A2 | R3 : + Pitot, vent, dérapage avec décalage | 0.19 | 0.10 | 0.10 | 0.97 | 1.13 | 0.68 | 0.50 |
| A2 | R3 sans état de décalage (dérapage σ 6°) | 0.20 | 0.11 | 0.17 | 1.03 | 2.45 | -2.17 | 1.10 |
| A2 | R3 sans décalage, dérapage σ 17° (valeur EKF2) | 0.18 | 0.10 | 0.10 | 0.93 | 0.98 | -0.52 | 0.67 |
| A2 | R3 sans dérapage | 0.19 | 0.10 | 0.09 | 0.94 | 0.95 | 0.41 | 0.47 |
| A2 | R3, magnétomètre 3 axes contre le WMM | 0.19 | 0.10 | 0.09 | 0.93 | 0.98 | 0.41 | 0.50 |

## Cohérence sur données réelles (NIS moyen par degré de liberté, en vol ; rejets au test à 99,9 %)

| Vol | Configuration | GNSS | Baro | Magnéto (cap) | Magnéto (3 axes) | Pitot | Dérapage |
|---|---|---|---|---|---|---|---|
| A1 | R3 : + Pitot, vent, dérapage avec décalage | 0.10 (0.0 %) | 0.01 (0.0 %) | 0.44 (0.0 %) | – | 0.17 (0.0 %) | 0.12 (0.0 %) |
| A1 | R3 sans état de décalage (dérapage σ 6°) | 0.20 (0.0 %) | 0.01 (0.0 %) | 0.46 (0.0 %) | – | 0.32 (0.0 %) | 0.41 (0.0 %) |
| A1 | R3, magnétomètre 3 axes contre le WMM | 0.10 (0.0 %) | 0.01 (0.0 %) | – | 0.14 (0.0 %) | 0.17 (0.0 %) | 0.12 (0.0 %) |
| A2 | R3 : + Pitot, vent, dérapage avec décalage | 0.30 (0.0 %) | 0.15 (0.0 %) | 0.30 (0.0 %) | – | 0.16 (0.0 %) | 0.09 (0.0 %) |
| A2 | R3 sans état de décalage (dérapage σ 6°) | 0.36 (0.0 %) | 0.16 (0.0 %) | 0.29 (0.0 %) | – | 0.30 (0.0 %) | 0.22 (0.0 %) |
| A2 | R3, magnétomètre 3 axes contre le WMM | 0.30 (0.0 %) | 0.16 (0.0 %) | – | 0.16 (0.0 %) | 0.16 (0.0 %) | 0.09 (0.0 %) |

## Calibrations estimées en vol

| Vol | Décalage de dérapage estimé [°] | ±1σ [°] | Dérapage déduit de l'EKF2 [°] | Échelle de vitesse air estimée | Rapport vitesse air / |v − w| de l'EKF2 |
|---|---|---|---|---|---|
| A1 | -8.3 | 0.3 | -9.6 | 0.992 | 0.983 |
| A2 | -6.6 | 0.2 | -7.1 | 1.000 | 1.000 |
| B | – | – | – | 1.352 ± 0.003 | 1.364 |

## Pertes GNSS simulées sur les vols réels (30 s, toutes les 90 s en vol)

Écart horizontal entre l'estimation à la fin de la perte et la première position GNSS qui suit.

| Configuration | Pertes | Écart médian [m] | Médiane A1 / A2 [m] | Écart max [m] | 1σ annoncé médian [m] | Écart / σ max |
|---|---|---|---|---|---|---|
| R1 : IMU + GNSS | 7 | 15.1 | 10.8 / 24.5 | 36.1 | 49.6 | 0.68 |
| R1 + baro | 7 | 23.8 | 14.8 / 36.9 | 54.5 | 37.9 | 1.35 |
| R2 : + baro, magnétomètre (cap) | 7 | 31.4 | 17.8 / 35.1 | 54.6 | 37.4 | 1.36 |
| R3 : + Pitot, vent, dérapage avec décalage | 7 | 12.0 | 12.0 / 11.7 | 19.1 | 16.0 | 1.19 |

## Provenance

- Empreinte du code : `c1649691a8762077` (16 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 0, 1 runs
- Paramètres : `9d47ac707f3dded2` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-03T08:49:50Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
