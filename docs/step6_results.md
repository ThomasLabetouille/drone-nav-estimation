# Étape 6 : résultats

Généré par `scripts/step6_degraded.py`. Mission de l'étape 5 (vent de 5 m/s), 40 runs par configuration, mêmes tirages pour toutes. Test d'innovation à 99.9 % quand il est actif. Erreurs de l'état délivré en temps réel.

## Perte GNSS de 180 à 240 s

| Configuration | Erreur horiz. après 30 s [m] | après 60 s [m] | pire run après 60 s [m] | 1σ annoncé après 60 s [m] | NEES position après 60 s | Mesures GNSS rejetées après retour |
|---|---|---|---|---|---|---|
| O1 : IMU + GNSS | 14.8 | 85.1 | 140.7 | 73.6 | 1.18 | 0.1 % |
| O2 : + magnétomètre et baro | 8.1 | 39.3 | 135.4 | 33.2 | 1.07 | 0.1 % |
| O3 : + Pitot et vent (étape 5) | 5.8 | 14.8 | 35.5 | 18.9 | 0.74 | 0.1 % |
| O4 : + vent quasi figé sans GNSS | 3.8 | 6.9 | 16.9 | 9.1 | 0.68 | 0.1 % |

## Vent pendant sa rotation (300 à 360 s), sans panne

| Configuration | Erreur de vent max [m/s] | NEES vent 300-370 s | Erreur de vent après 400 s [m/s] | Pitot rejeté 300-400 s | Réinitialisations du vent |
|---|---|---|---|---|---|
| O3 : marche aléatoire 0,1 (étape 5) | 0.17 | 0.6 | 0.12 | 0.1 % | 0 |
| W1 : marche aléatoire 0,01, sans protection | 3.43 | 844.8 | 1.86 | 73.0 % | 0 |
| W2 : marche aléatoire 0,01, avec protection | 1.83 | 167.7 | 0.04 | 10.0 % | 40 |

## Saut GNSS de 15 m vers l'est, de 320 à 350 s

| Configuration | Erreur horiz. max pendant le saut [m] | NEES position pendant le saut | Erreur horiz. max après le saut [m] | Mesures GNSS rejetées pendant le saut | Mesures GNSS rejetées après le saut | Réinitialisations sur le GNSS |
|---|---|---|---|---|---|---|
| J1 : sans test d'innovation | 4.4 | 7.5 | 3.9 | 0.0 % | 0.0 % | 0 |
| J2 : avec test, vent de l'étape 5 | 7.7 | 1.2 | 3.8 | 99.9 % | 1.6 % | 1 |
| J3 : avec test, vent quasi figé sans GNSS, sans réinitialisation | 43.5 | 40.1 | 413.8 | 100.0 % | 100.0 % | 0 |
| J4 : comme J3, réinitialisation sur le GNSS après 45 s | 43.5 | 40.1 | 84.5 | 100.0 % | 11.6 % | 40 |
| J5 : comme J2, réinitialisation après 10 s | 15.3 | 49.1 | 15.2 | 33.4 % | 7.8 % | 80 |

## Perturbation magnétique de 170 à 230 s

| Configuration | Erreur de cap max [°] | NEES attitude pendant | Mesures magnéto rejetées pendant | Mesures magnéto rejetées hors panne |
|---|---|---|---|---|
| M1 : perturbation de 0,058 G, sans test | 7.75 | 9981.9 | 0.0 % | 0.00 % |
| M2 : perturbation de 0,058 G, avec test | 0.51 | 1.1 | 100.0 % | 0.10 % |
| M3 : perturbation de 0,009 G, avec test | 1.50 | 288.8 | 1.6 % | 0.10 % |

Vol animé (`docs/img/step6_outage.mp4`) : run 32, celui dont l'erreur de O3 à la fin de la perte est la médiane.

## Provenance

- Empreinte du code : `26de7852b0075eb4` (14 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 2026, 40 runs
- Paramètres : `cfea53456d61250d` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-04T15:57:02Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
