# Étape 1 : résultats

Généré par `scripts/step1_altitude.py`, 50 runs Monte-Carlo par scénario, graine 2026. Mission de 300 s.

| | A | B | C |
|---|---|---|---|
| Dérive baro dans la simulation | non | oui (σ = 1.5 m) | oui |
| Dérive baro dans le filtre | non | non | oui |
| RMSE altitude, baro brut [m] | 0.40 | 1.46 | 1.46 |
| RMSE altitude, Kalman [m] | 0.07 | 1.41 | 1.40 |
| RMSE Vz, vario baro filtré [m/s] | 0.75 | 0.76 | 0.76 |
| RMSE Vz, Kalman [m/s] | 0.036 | 0.097 | 0.074 |
| NEES moyen [h, v] (attendu : 2) | 1.98 | 967 | 1.97 |
| Temps passé dans l'intervalle NEES 95 % | 95% | 0% | 98% |
| NIS moyen du baro (attendu : 1) | 1.00 | 1.17 | 1.00 |
| Fenêtres de 10 s où le NIS dépasse son seuil à 97,5 % | 2.3% | 41.4% | 2.2% |

Accéléromètre seul, intégré depuis l'état vrai : erreur moyenne de 76 m à 60 s et 1898 m à 300 s.
