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
| A1 | vol complet | 704 | 150630 | 55.7 | 0.69 | 0.59 | 3.91 |
| A1 | pertes GNSS de 30 s | 704 | 150630 | 54.0 | 0.67 | 0.57 | 3.79 |
| A2 | vol complet | 635 | 135878 | 47.9 | 0.62 | 0.53 | 3.91 |
| A2 | pertes GNSS de 30 s | 635 | 135878 | 49.4 | 0.62 | 0.54 | 3.94 |
| B | vol complet | 592 | 131419 | 45.4 | 0.66 | 0.55 | 4.16 |
| B | pertes GNSS de 30 s | 592 | 131419 | 46.1 | 0.64 | 0.52 | 3.96 |

Tests unitaires C++ (`cpp/tests/test_eskf.cpp`) : predict 3.44 us, GNSS update 9.50 us (20 states, this machine).

## Provenance

- Empreinte du code : `577ab746e3aebb55` (24 fichiers, calculée par le script au moment de l'écriture)
- Commit git : non disponible
- Graine : 0, 1 runs
- Paramètres : `2a356de0c52cd507` (détail dans le fichier JSON voisin)
- Python 3.13.15, numpy 2.5.3, scipy 1.18.1
- Généré le 2026-10-04T13:16:57Z

`python scripts/check_provenance.py` compare cette empreinte au code actuel.
