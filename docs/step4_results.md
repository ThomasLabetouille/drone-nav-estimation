# Étape 4 : résultats

Généré par `scripts/step4_gnss.py`. Mission de 480 s, 50 runs par configuration, même graine pour toutes. Erreurs RMS de l'état délivré en temps réel, NEES et NIS par degré de liberté (attendu : 1), sur t > 120 s.

GNSS réaliste : bruit blanc 0.5 m / 1.0 m, erreur corrélée 1.4 m / 2.8 m (τ = 60 s), latence 150 ms.

## Erreurs corrélées, sans latence

| Configuration | Position horiz. [m] | Position vert. [m] | Vitesse horiz. [m/s] | Inclinaison [°] | Cap [°] | NEES pos. | NEES att. | NEES biais gyro | NIS |
|---|---|---|---|---|---|---|---|---|---|
| A : erreurs blanches | 0.24 | 0.35 | 0.048 | 0.064 | 0.50 | 1.02 | 1.05 | 1.01 | 1.00 |
| B : erreurs corrélées, traitées comme blanches | 1.78 | 2.45 | 0.049 | 0.064 | 0.46 | 53.02 | 1.01 | 0.96 | 0.65 |
| C : erreurs corrélées, estimées (18 états) | 1.18 | 1.68 | 0.048 | 0.063 | 0.46 | 0.93 | 0.99 | 0.96 | 1.00 |

## Latence de 150 ms (erreurs corrélées estimées)

| Configuration | Position horiz. [m] | Position vert. [m] | Vitesse horiz. [m/s] | Inclinaison [°] | Cap [°] | NEES pos. | NEES att. | NEES biais gyro | NIS |
|---|---|---|---|---|---|---|---|---|---|
| latence ignorée | 2.97 | 1.55 | 0.422 | 0.262 | 2.63 | 4.23 | 61.41 | 6.29 | 1.18 |
| latence compensée sur la position | 2.34 | 1.55 | 0.415 | 0.255 | 2.57 | 2.67 | 58.38 | 6.03 | 1.25 |
| horizon retardé + prédicteur de sortie | 1.23 | 1.55 | 0.051 | 0.065 | 0.46 | 0.94 | 1.02 | 1.03 | 1.00 |

## Provenance

- Empreinte du code : `8f628191505510b3` (11 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 2026, 50 runs
- Paramètres : `8d7f815c02b74084` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-02T13:51:29Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
