# Étape 2 : résultats

Généré par `scripts/step2_strapdown.py`. Mission de 480 s, IMU à 200 Hz.

## Strapdown avec une IMU parfaite

| Variante | Erreur de position à la fin [m] | Erreur d'attitude max [°] |
|---|---|---|
| naïve | 1.14 | 5.2e-05 |
| + compensation de rotation | 0.291 | 5.2e-05 |
| + coning et sculling | 0.00836 | 2.7e-08 |

## Navigation à l'estime, IMU MEMS, vol rectiligne (500 runs)

| Temps | RMS horizontal par axe [m] | Budget [m] | Rapport | RMS vertical [m] | Budget vertical [m] |
|---|---|---|---|---|---|
| 10 s | 2.7 | 2.6 | 1.04 | 2.5 | 2.5 |
| 30 s | 28.6 | 27.5 | 1.04 | 22.1 | 22.6 |
| 60 s | 159.8 | 154.5 | 1.03 | 88.0 | 90.2 |
| 120 s | 1103.9 | 1077.2 | 1.02 | 351.6 | 360.8 |

Avec 500 runs, un rapport entre 0.94 et 1.06 est compatible avec le budget (intervalle à 95 % d'une RMS estimée).

## Provenance

- Empreinte du code : `1ab896b627b90d37` (8 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 2026, 500 runs
- Paramètres : `7b3fcc9bcc4abf4c` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-02T10:24:38Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
