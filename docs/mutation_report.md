# Tests de mutation

Généré par `scripts/mutation_check.py`. Chaque ligne est un bug plausible introduit dans une copie du dépôt, puis la suite de tests est lancée sur cette copie (sans le test de provenance, qui détecterait trivialement toute modification).

**23 mutations détectées sur 25.**

| | Fichier | Bug simulé | Résultat | Détecté par |
|---|---|---|---|---|
| M01 | `eskf.py` | signe du couplage vitesse/attitude dans F | détecté | `tests/test_step3.py::test_transition_matrix_matches_nonlinear_propagation[30.0]` |
| M02 | `eskf.py` | signe du couplage attitude/biais gyro dans F | détecté | `tests/test_step3.py::test_transition_matrix_matches_nonlinear_propagation[30.0]` |
| M03 | `eskf.py` | bruit de processus gyro en dt² au lieu de dt | détecté | `tests/test_oracles.py::test_propagated_covariance_matches_monte_carlo[noise_only]` |
| M04 | `eskf.py` | correction d'attitude injectée dans le repère avion au lieu de NED | détecté | `tests/test_step3.py::test_filter_is_consistent_on_a_short_flight` |
| M05 | `eskf.py` | signe de la jacobienne de réinitialisation | survit (attendu) | `` |
| M06 | `eskf.py` | forme de Joseph remplacée par la forme courte (I - KH) P | survit (attendu) | `` |
| M07 | `eskf.py` | biais GNSS oublié dans le modèle de mesure | détecté | `tests/test_step4.py::test_gnss_bias_states_follow_gauss_markov` |
| M08 | `strapdown.py` | signe de la correction de coning | détecté | `tests/test_step2.py::test_perfect_imu_reproduces_mission` |
| M09 | `strapdown.py` | gravité de signe inversé en NED | détecté | `tests/test_step2.py::test_perfect_imu_reproduces_mission` |
| M10 | `strapdown.py` | signe de la compensation de rotation | détecté | `tests/test_step2.py::test_perfect_imu_reproduces_mission` |
| M11 | `strapdown.py` | erreur dans le budget d'erreur analytique | détecté | `tests/test_step2.py::test_dead_reckoning_matches_error_budget` |
| M12 | `rotations.py` | un terme du produit de quaternions | détecté | `tests/test_oracles.py::TestRotationsAgainstScipy::test_product_is_composition` |
| M13 | `rotations.py` | convention des angles d'Euler | détecté | `tests/test_oracles.py::TestRotationsAgainstScipy::test_euler_to_quaternion` |
| M14 | `trajectory3d.py` | loi du virage coordonné | détecté | `tests/test_step2.py::test_specific_force_in_steady_turn` |
| M15 | `trajectory3d.py` | conversion vitesses d'Euler vers vitesses angulaires | détecté | `tests/test_oracles.py::test_trajectory_matches_ode_integration` |
| M16 | `imu.py` | densité de bruit appliquée en dt au lieu de √dt | détecté | `tests/test_oracles.py::test_propagated_covariance_matches_monte_carlo[noise_only]` |
| M17 | `sensors.py` | discrétisation du processus de Gauss-Markov | détecté | `tests/test_step1.py::test_filter_is_consistent_when_model_matches` |
| M18 | `gnss.py` | temps de corrélation GNSS en mauvaise unité | détecté | `tests/test_step4.py::test_correlated_error_statistics` |
| M19 | `kf_altitude.py` | dérive baro absente du modèle de mesure | détecté | `tests/test_step1.py::test_filter_is_consistent_when_model_matches` |
| M20 | `kf_altitude.py` | formule de Van Loan incomplète | détecté | `tests/test_oracles.py::test_van_loan_matches_covariance_ode` |
| M21 | `kf_altitude.py` | signe du terme croisé de la covariance initiale (étape 1) | détecté | `tests/test_step1.py::test_filter_is_consistent_when_model_matches` |
| M22 | `fusion.py` | compensation de latence dans le mauvais sens | détecté | `tests/test_oracles.py::test_latency_geometry[compensate-0.0]` |
| M23 | `fusion.py` | signe du terme croisé de la covariance initiale (étape 4) | détecté | `tests/test_step4.py::test_initial_covariance_correlates_position_and_gnss_bias` |
| M24 | `fusion.py` | décalage d'un échantillon IMU (5 ms) dans l'horizon retardé | détecté | `tests/test_step4.py::test_delayed_mode_without_latency_equals_plain_filter` |
| M25 | `provenance.py` | fins de ligne non normalisées avant le hachage | détecté | `tests/test_provenance.py::test_fingerprint_ignores_line_endings` |

## Mutations qui survivent, et pourquoi

- **M05** (signe de la jacobienne de réinitialisation) : effet du second ordre : la correction d'attitude injectée fait quelques millièmes de radian, donc la réinitialisation modifie P de moins de 0,1 %.
- **M06** (forme de Joseph remplacée par la forme courte (I - KH) P) : mutant équivalent : avec le gain optimal, les deux formes sont égales en arithmétique exacte. La forme de Joseph n'apporte que de la robustesse numérique.

## Provenance

- Empreinte des sources et des tests : `a0abcd63c10ecad1` (22 fichiers)
- Toute modification d'un module ou d'un test rend ce rapport périmé (`scripts/check_provenance.py`) : relancer le script.
