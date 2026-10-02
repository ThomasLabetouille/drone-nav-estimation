# Étape 3 : résultats

Généré par `scripts/step3_eskf.py`. Mission de 480 s, IMU à 200 Hz, GNSS à 5 Hz, 100 runs. Premier virage à 71 s.

## Erreurs RMS après convergence (t > 120 s)

| | Nord | Est | Bas |
|---|---|---|---|
| Position [m] | 0.17 | 0.17 | 0.34 |
| Vitesse [m/s] | 0.034 | 0.034 | 0.044 |

| | Inclinaison autour du nord | Inclinaison autour de l'est | Cap |
|---|---|---|---|
| Attitude [°] | 0.064 | 0.064 | 0.43 |

## Cohérence

| Bloc | NEES ou NIS moyen / ddl (attendu : 1) | Temps dans l'intervalle à 95 % |
|---|---|---|
| Position | 1.00 | 98% |
| Vitesse | 1.00 | 94% |
| Attitude | 1.01 | 86% |
| Biais gyro | 1.02 | 87% |
| Biais accéléro | 1.03 | 99% |
| NIS GNSS | 1.00 | 95% |

## Cohérence de l'attitude selon l'erreur de cap initiale

| σ du cap initial | NEES attitude / ddl, de 5 s au premier virage | après 100 s |
|---|---|---|
| 5° | 1.20 | 0.97 |
| 2° | 1.07 | 1.01 |
| 1° | 1.02 | 1.01 |

## Observabilité : incertitude moyenne (1σ) avant et après les manœuvres

| État | Avant le premier virage (t = 65 s) | Après (t = 90 s) | Fin de mission |
|---|---|---|---|
| Cap [°] | 4.50 | 0.29 | 0.36 |
| Inclinaison [°] | 0.294 | 0.063 | 0.065 |
| Biais accéléro x [m/s²] | 0.0394 | 0.0079 | 0.0064 |
| Biais accéléro z [m/s²] | 0.0061 | 0.0035 | 0.0033 |

| État | Avant les deux boucles (t = 105 s) | Après (t = 160 s) | Fin de mission |
|---|---|---|---|
| Biais gyro z [°/s] | 0.0199 | 0.0072 | 0.0095 |

## Provenance

- Empreinte du code : `cba250afe43d20af` (13 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 2026, 100 runs
- Paramètres : `b121f9442747f2a7` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-02T16:39:50Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
