# drone-nav-estimation

[![tests](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml)

Simulation de capteurs et estimation d'état pour un drone à voilure fixe, en Python, avec un portage C++ prévu une fois les algorithmes validés.

Le projet avance par étapes. Chacune ajoute des capteurs ou des états au filtre et se valide avant de passer à la suivante. L'étape 1 est faite : la voie verticale, soit un filtre de Kalman qui fusionne un accéléromètre et un baromètre.

![Estimation d'altitude sur une mission complète](docs/img/step1_estimation.png)

## Lancer

```bash
pip install -e ".[dev]"
python -m pytest                     # une dizaine de secondes
python scripts/step1_altitude.py     # figures + tableau de résultats, ~45 s
python scripts/step1_altitude.py --runs 10   # plus rapide, moins de Monte-Carlo
```

Les figures sont écrites dans `docs/img/`, le tableau dans `docs/step1_results.md`.

## Étape 1 : altitude, accéléromètre + baromètre

### Mission simulée

300 s de vol : 10 s immobile sur la catapulte, montée à 120 m, palier, montée à 150 m, palier, descente à 30 m. Les transitions sont des polynômes de degré 5 (jerk minimal), donc l'accélération reste continue. La turbulence verticale est une somme de trois sinusoïdes (périodes de 2 à 12 s, jusqu'à ~3 m/s²). La vérité terrain est analytique, ce qui garantit que `a = dv/dt` et `v = dh/dt` exactement. Un test le vérifie.

### Capteurs

| | Fréquence | Modèle d'erreur |
|---|---|---|
| Accéléromètre (axe vertical) | 100 Hz | bruit blanc 0,02 m/s²/√Hz, biais initial σ = 0,05 m/s² (≈ 5 mg), marche aléatoire 5·10⁻⁴ m/s²/√s |
| Baromètre | 25 Hz | bruit blanc σ = 0,4 m, dérive lente Gauss-Markov σ = 1,5 m, τ = 120 s |

La densité de bruit de l'accéléro est volontairement 10 à 20 fois au-dessus d'une fiche technique MEMS, pour représenter les vibrations moteur. La dérive du baro représente la météo, la température et l'erreur de prise de pression statique.

Simplification de cette étape : l'accéléro mesure directement l'accélération verticale, gravité retirée et attitude supposée parfaite. Le modèle strapdown complet arrive à l'étape 2.

### Filtre

État `[h, v, b_a]`, plus `d` (dérive baro) quand le filtre la modélise. L'accéléro sert d'entrée de commande à la prédiction, le baro sert de mesure (`z = h + d + bruit`).

- Le modèle est écrit en temps continu puis discrétisé exactement (méthode de Van Loan). Le réglage du filtre reste donc exprimé en unités physiques (densités de bruit, marches aléatoires) et ne dépend pas de la fréquence d'échantillonnage.
- La mise à jour se fait en forme de Joseph.
- À l'initialisation sur le premier point baro, l'erreur d'altitude vaut `bruit + d` et l'erreur sur `d` vaut `-d`. La covariance initiale contient donc un terme croisé négatif. Sans lui, le filtre est incohérent pendant les premières secondes.

Le code est dans `src/navsim/kf_altitude.py`.

### Trois scénarios

- **A** : baro sans dérive, filtre à 3 états. C'est le cas de référence, où le modèle du filtre correspond exactement à la simulation.
- **B** : baro avec dérive, filtre à 3 états qui l'ignore.
- **C** : baro avec dérive, filtre à 4 états qui la modélise.

### Résultats (50 runs Monte-Carlo par scénario)

| | A | B | C |
|---|---|---|---|
| RMSE altitude, baro brut [m] | 0,40 | 1,46 | 1,46 |
| RMSE altitude, Kalman [m] | 0,07 | 1,41 | 1,40 |
| RMSE Vz, vario baro filtré (τ = 1 s) [m/s] | 0,75 | 0,76 | 0,76 |
| RMSE Vz, Kalman [m/s] | 0,036 | 0,097 | 0,074 |
| NEES moyen sur [h, v] (attendu : 2) | 1,98 | 967 | 1,97 |
| Temps dans l'intervalle NEES à 95 % | 95 % | 0 % | 98 % |
| Fenêtres de 10 s où le NIS baro dépasse son seuil | 2,3 % | 41 % | 2,2 % |

![Pourquoi fusionner](docs/img/step1_why_fusion.png)

L'accéléromètre seul, intégré deux fois depuis l'état vrai, dérive de 76 m en 60 s et de près de 1,9 km en 300 s. Il suffit d'un biais de quelques mg. Le baro seul ne dérive pas autant, mais il est bruité, et sa dérivée est inutilisable sans filtrage, ce qui ajoute du retard. Le Kalman garde la dynamique de l'accéléro et la référence absolue du baro. Il estime aussi le biais de l'accéléro en ligne : en une trentaine de secondes, l'incertitude sur le biais passe de 0,05 à moins de 0,005 m/s².

![Cohérence du filtre](docs/img/step1_consistency.png)

Avec une dérive du baro, l'altitude ne peut pas être meilleure que celle du baro, que le filtre la modélise ou non (1,40 m contre 1,41 m). Tant que le baro est la seule référence absolue, la dérive n'est pas observable. La différence entre B et C se joue sur la covariance :

- En B, le filtre annonce une incertitude de quelques centimètres alors que l'erreur réelle fait plus d'un mètre. Le NEES est à ~1000 au lieu de 2. Un filtre dans cet état rejetterait à tort une bonne mesure GNSS qui contredit le baro.
- En C, l'erreur est la même, mais la covariance la couvre (NEES ≈ 2).
- Sans vérité terrain, donc en vol réel, le seul indicateur est le NIS du baro. En B, il ne dépasse son seuil que dans 41 % des fenêtres de 10 s : sur un seul vol, la dérive non modélisée passe facilement inaperçue.

La seule façon d'observer cette dérive est d'avoir une deuxième référence d'altitude. C'est le rôle du GNSS dans les étapes suivantes.

### Écarts rencontrés

Dans une première version, l'accéléromètre simulé échantillonnait `a(t_k)`, alors que le filtre suppose une accélération constante sur `[t_k, t_k+1]`. Le scénario A, censé être parfaitement cohérent, donnait un NEES moyen de 2,09, avec 92 % du temps dans l'intervalle à 95 %. Le filtre était donc légèrement optimiste. J'ai corrigé en faisant sortir à l'accéléro simulé l'accélération moyenne sur la période (`Δv / dt`), comme le font les IMU qui fournissent des incréments de vitesse. Le NEES est alors passé à 1,98, avec 95 % du temps dans l'intervalle.

## Organisation

```
src/navsim/
  trajectory.py     trajectoire de référence (vérité terrain)
  sensors.py        modèles d'erreur capteurs
  kf_altitude.py    filtre de Kalman de la voie verticale, discrétisation Van Loan
  runner.py         boucle de simulation, NEES/NIS, navigation à l'estime
scripts/
  step1_altitude.py figures et tableau de l'étape 1
tests/
  test_step1.py     vérité cohérente, Van Loan, covariance définie positive, cohérence NEES
docs/
  step1_results.md  tableau généré par le script
  img/              figures
```

## Suite

1. Trajectoire 3D voilure fixe (virages coordonnés, cercle d'attente) et IMU 6 axes. On vérifie d'abord qu'une IMU parfaite réintégrée redonne la trajectoire.
2. EKF à état d'erreur à 15 états (position, vitesse, attitude en quaternion, biais gyro, biais accéléro) avec le GNSS.
3. GNSS retardé (100 à 200 ms) fusionné à l'horizon retardé avec un buffer d'état.
4. Baro, magnétomètre, Pitot et estimation du vent.
5. Modes dégradés : perte GNSS, perturbation magnétique, vibrations. Gating des innovations et détection de capteur défaillant.
6. Rejeu de logs de vol PX4 réels et comparaison avec l'EKF2 embarqué.
7. Portage C++ (matrices de taille fixe, sans allocation dynamique) et comparaison avec la référence Python sur les mêmes logs.
