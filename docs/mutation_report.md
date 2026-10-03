# Tests de mutation

Généré par `scripts/mutation_check.py`. Chaque ligne est un bug plausible introduit dans une copie du dépôt, puis la suite de tests est lancée sur cette copie (sans le test de provenance, qui détecterait trivialement toute modification).

**40 mutations détectées sur 41.**

| | Fichier | Bug simulé | Résultat | Détecté par |
|---|---|---|---|---|
| M01 | `eskf.py` | signe du couplage vitesse/attitude dans F | détecté | `tests/test_step3.py::test_transition_matrix_matches_nonlinear_propagation[30.0]` |
| M02 | `eskf.py` | signe du couplage attitude/biais gyro dans F | détecté | `tests/test_step3.py::test_transition_matrix_matches_nonlinear_propagation[30.0]` |
| M03 | `eskf.py` | bruit de processus gyro en dt² au lieu de dt | détecté | `tests/test_oracles.py::test_propagated_covariance_matches_monte_carlo[noise_only]` |
| M04 | `eskf.py` | correction d'attitude injectée dans le repère avion au lieu de NED | détecté | `tests/test_step3.py::test_filter_is_consistent_on_a_short_flight` |
| M05 | `eskf.py` | signe de la jacobienne de réinitialisation (convention d'erreur locale au lieu de globale) : bug réel du projet jusqu'à l'étape 5 | détecté | `tests/test_oracles.py::test_reset_jacobian_matches_rotation_composition[0]` |
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
| M26 | `eskf.py` | signe de la jacobienne magnétomètre / attitude | détecté | `tests/test_step5.py::test_measurement_jacobian_matches_numerical_derivative[0-mag]` |
| M27 | `eskf.py` | signe du vent dans la mesure Pitot | détecté | `tests/test_step5.py::test_measurement_jacobian_matches_numerical_derivative[0-airspeed]` |
| M28 | `eskf.py` | dérivée de 1/V oubliée dans la jacobienne du dérapage (erreur faite puis corrigée pendant l'écriture) | détecté | `tests/test_step5.py::test_measurement_jacobian_matches_numerical_derivative[0-sideslip]` |
| M29 | `eskf.py` | altitude baro prise égale à +p_D au lieu de -p_D | détecté | `tests/test_step5.py::test_measurement_jacobian_matches_numerical_derivative[0-baro]` |
| M30 | `fusion.py` | biais magnétomètre appris en ligne droite au lieu des virages | détecté | `tests/test_step5.py::test_gated_mag_bias_keeps_the_heading_consistent` |
| M31 | `trajectory3d.py` | accélération du vent absente de la force spécifique | détecté | `tests/test_step5.py::test_changing_wind_is_in_the_specific_force` |
| M32 | `aiding.py` | marche aléatoire du biais magnétomètre en mauvaise unité | détecté | `tests/test_step5.py::test_magnetometer_stream_statistics` |
| M33 | `aiding.py` | signe du cap magnétique | détecté | `tests/test_step5.py::test_magnetometer_heading_recovers_the_true_heading` |
| M34 | `fusion.py` | signe de la corrélation vent / cap dans la covariance initiale du vent | détecté | `tests/test_step5.py::test_initial_wind_covariance_matches_monte_carlo` |
| M35 | `eskf.py` | test d'innovation inversé | détecté | `tests/test_step6.py::test_gate_rejects_outliers_and_leaves_the_filter_untouched` |
| M36 | `eskf.py` | mesure rejetée mais fusionnée quand même | détecté | `tests/test_step6.py::test_gate_rejects_outliers_and_leaves_the_filter_untouched` |
| M37 | `fusion.py` | degrés de liberté du test GNSS (position seule au lieu de position + vitesse) | détecté | `tests/test_step6.py::test_gnss_false_alarm_rate_in_flight` |
| M38 | `fusion.py` | protection contre le blocage du Pitot désactivée | détecté | `tests/test_step6.py::test_wind_lockout_and_its_protection` |
| M39 | `fusion.py` | marche aléatoire du vent réduite avec GNSS au lieu de sans | détecté | `tests/test_step6.py::test_gnss_lockout_and_its_protection` |
| M40 | `fusion.py` | perte GNSS simulée ignorée | détecté | `tests/test_step6.py::test_outage_is_applied_and_gnss_is_accepted_again` |
| M41 | `aiding.py` | perturbation magnétique simulée ignorée | détecté | `tests/test_step6.py::test_strong_magnetic_disturbance_is_rejected` |

## Mutations qui survivent, et pourquoi

- **M06** (forme de Joseph remplacée par la forme courte (I - KH) P) : mutant équivalent : avec le gain optimal, les deux formes sont égales en arithmétique exacte. La forme de Joseph n'apporte que de la robustesse numérique.

## Provenance

- Empreinte des sources et des tests : `5a2300792eabb998` (26 fichiers)
- Toute modification d'un module ou d'un test rend ce rapport périmé (`scripts/check_provenance.py`) : relancer le script.
