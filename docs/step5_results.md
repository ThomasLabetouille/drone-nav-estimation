# Étape 5 : résultats

Généré par `scripts/step5_aiding.py`. Mission de 480 s dans un vent de 5 m/s d'ouest qui tourne au nord-ouest (6 m/s) entre 300 et 360 s ; écart initial entre route et cap : 16.4°. GNSS réaliste, biais GNSS estimé, horizon retardé. 40 runs par configuration, mêmes tirages IMU et GNSS pour toutes. Premier virage à 71 s.

## Cap, position, cohérence

| Configuration | Cap avant 1er virage [°] | Cap après 120 s [°] | NEES att. avant | NEES att. après | Position horiz. [m] | Position vert. [m] |
|---|---|---|---|---|---|---|
| E1 : GNSS seul, cap initial = route GNSS | 16.11 | 0.56 | 4.72 | 1.05 | 1.26 | 1.64 |
| E2 : + magnétomètre | 2.28 | 0.10 | 0.97 | 0.99 | 1.27 | 1.64 |
| E2b : magnétomètre, biais appris en permanence | 2.58 | 0.10 | 1.43 | 1.01 | 1.27 | 1.64 |
| E3 : + baromètre | 2.28 | 0.10 | 0.96 | 0.99 | 1.27 | 1.02 |
| E4 : + Pitot et vent | 2.26 | 0.10 | 0.99 | 0.99 | 1.27 | 1.02 |
| E5a : + dérapage nul, σ 2° (naïf) | 2.28 | 0.10 | 1.00 | 1.02 | 1.27 | 1.00 |
| E5i : σ 6°, vent initial nul | 3.12 | 0.10 | 1.39 | 1.00 | 1.27 | 1.00 |
| E5 : + dérapage nul, σ 6° | 2.27 | 0.10 | 1.00 | 1.00 | 1.27 | 1.00 |

## États ajoutés (après 120 s, sauf mention)

| Configuration | NEES biais magnéto avant 1er virage | NEES biais magnéto | NEES dérive baro | NEES vent | Vent le long [m/s] | Vent de travers [m/s] | Vent de travers avant 1er virage [m/s] | NIS magnéto | NIS baro | NIS Pitot |
|---|---|---|---|---|---|---|---|---|---|---|
| E2 : + magnétomètre | 1.02 | 1.08 | – | – | – | – | – | 1.00 | – | – |
| E2b : magnétomètre, biais appris en permanence | 2.87 | 1.18 | – | – | – | – | – | 1.00 | – | – |
| E3 : + baromètre | 1.02 | 1.08 | 0.93 | – | – | – | – | 1.00 | 1.00 | – |
| E4 : + Pitot et vent | 1.03 | 1.08 | 0.93 | 0.36 | 0.08 | 0.11 | 0.68 | 1.00 | 1.00 | 0.96 |
| E5a : + dérapage nul, σ 2° (naïf) | 1.04 | 1.10 | 0.91 | 1.18 | 0.10 | 0.18 | 0.67 | 1.00 | 1.00 | 0.99 |
| E5i : σ 6°, vent initial nul | 1.34 | 1.09 | 0.91 | 0.41 | 0.08 | 0.07 | 0.93 | 1.00 | 1.00 | 0.96 |
| E5 : + dérapage nul, σ 6° | 1.03 | 1.08 | 0.91 | 0.41 | 0.08 | 0.07 | 0.67 | 1.00 | 1.00 | 0.96 |

## Provenance

- Empreinte du code : `4be25335bd5fa137` (13 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 2026, 40 runs
- Paramètres : `816840ae16ec651a` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-02T17:09:18Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
