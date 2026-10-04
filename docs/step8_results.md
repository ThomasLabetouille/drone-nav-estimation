# Étape 8 : résultats

Généré par `scripts/step8_cpp.py`. Configuration R3 de l'étape 7, mêmes logs. Écarts maximaux sur toutes les époques GNSS du vol.

## Concordance

| Vol | Rejeu | Époques | Position [m] | Vitesse [m/s] | Attitude [rad] | États ajoutés | σ (relatif) | Mises à jour | Décisions du test différentes |
|---|---|---|---|---|---|---|---|---|---|
| A1 | vol complet | 1408 | 5.7e-14 | 3.2e-14 | 5.6e-15 | 7.8e-14 | 5.2e-14 | 9516 | 0 |
| A1 | pertes GNSS de 30 s | 1408 | 8.7e-13 | 7.8e-14 | 5.6e-15 | 7.8e-14 | 5.2e-14 | 9336 | 0 |
| A2 | vol complet | 1270 | 5.7e-14 | 2.0e-14 | 3.3e-15 | 2.3e-13 | 5.6e-14 | 9828 | 0 |
| A2 | pertes GNSS de 30 s | 1270 | 1.5e-12 | 6.3e-14 | 3.5e-15 | 2.8e-13 | 5.0e-14 | 9587 | 0 |
| B | vol complet | 4462 | 3.7e-13 | 7.3e-14 | 1.1e-14 | 2.9e-13 | 2.0e-12 | 11782 | 0 |
| B | pertes GNSS de 30 s | 4462 | 9.2e-12 | 7.9e-13 | 1.3e-14 | 3.8e-13 | 2.0e-12 | 10382 | 0 |

## Temps de calcul (cette machine)

| Vol | Rejeu | Durée rejouée [s] | Événements | Python [s] | C++, programme complet [s] | C++, filtre seul [s] | C++, par événement [µs] |
|---|---|---|---|---|---|---|---|
| A1 | vol complet | 704 | 150630 | 52.8 | 0.69 | 0.58 | 3.86 |
| A1 | pertes GNSS de 30 s | 704 | 150630 | 53.6 | 0.67 | 0.57 | 3.76 |
| A2 | vol complet | 635 | 135878 | 48.1 | 0.68 | 0.59 | 4.34 |
| A2 | pertes GNSS de 30 s | 635 | 135878 | 48.8 | 0.63 | 0.55 | 4.04 |
| B | vol complet | 592 | 131419 | 44.8 | 0.65 | 0.54 | 4.14 |
| B | pertes GNSS de 30 s | 592 | 131419 | 43.6 | 0.63 | 0.53 | 4.00 |

Tests unitaires C++ (`cpp/tests/test_eskf.cpp`) : predict 3.71 us, GNSS update 9.79 us (20 states, this machine).

## Provenance

- Empreinte du code : `a170d079ab6e7bd6` (24 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 0, 1 runs
- Paramètres : `2a356de0c52cd507` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-04T16:21:22Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
