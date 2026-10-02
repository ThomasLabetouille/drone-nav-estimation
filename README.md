# drone-nav-estimation

[![tests](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml)

Simulation de capteurs et estimation d'état pour un drone à voilure fixe, en Python, avec un portage C++ prévu une fois les algorithmes validés.

Le projet avance par étapes. Chacune ajoute des capteurs ou des états au filtre et se valide avant de passer à la suivante. Deux étapes sont faites :

1. la voie verticale : un filtre de Kalman qui fusionne un accéléromètre et un baromètre ;
2. la navigation inertielle en 3D : trajectoire de voilure fixe, IMU 6 axes, intégration strapdown, et l'erreur d'une IMU MEMS seule comparée à son budget analytique.

![Estimation d'altitude sur une mission complète](docs/img/step1_estimation.png)

## Lancer

```bash
pip install -e ".[dev]"
python -m pytest                       # environ une minute
python scripts/step1_altitude.py       # étape 1, ~45 s
python scripts/step2_strapdown.py      # étape 2, ~50 s
python scripts/step2_strapdown.py --runs 100   # plus rapide, moins de Monte-Carlo
python scripts/check_provenance.py     # les résultats de docs/ correspondent-ils au code actuel ?
```

Les figures sont écrites dans `docs/img/`, les tableaux dans `docs/stepN_results.md`.

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

## Étape 2 : trajectoire 3D, IMU 6 axes, navigation inertielle

![Mission de 8 minutes](docs/img/step2_trajectory.png)

### Trajectoire et IMU idéale

Modèle cinématique, sans vent ni dérapage. Trois consignes, chacune construite avec les mêmes transitions à jerk minimal qu'à l'étape 1 : l'inclinaison, la pente et la vitesse air. Le cap suit la loi du virage coordonné, `dψ/dt = g tan φ / V`, et l'assiette vaut la pente plus un angle de calage constant de 3°.

Les vitesses angulaires et la force spécifique dans le repère avion ne dépendent pas du cap. Elles sont donc analytiques. Seuls le cap et la position demandent une intégration numérique, faite par la méthode de Simpson sur une grille 8 fois plus fine que l'IMU (1600 Hz).

L'IMU tourne à 200 Hz et sort des incréments, comme une vraie IMU : un incrément d'angle (intégrale de la vitesse angulaire) et un incrément de vitesse (intégrale de la force spécifique), tous deux dans le repère avion. Terre plate, repère NED, gravité constante, sans rotation terrestre. Le strapdown fait les mêmes hypothèses. Il faudra y revenir pour rejouer des logs réels.

Le code est dans `src/navsim/trajectory3d.py`, `src/navsim/rotations.py` (quaternions, conventions PX4) et `src/navsim/imu.py`.

### Strapdown, et ce qu'il reste avec une IMU parfaite

`src/navsim/strapdown.py` intègre les incréments selon trois variantes. Avec une IMU parfaite, l'erreur restante est celle de l'algorithme seul :

| Variante | Position après 8 min | Attitude, erreur max |
|---|---|---|
| naïve : `v += C(q) Δv` | 1,14 m | 5·10⁻⁵ ° |
| + compensation de rotation `½ Δθ × Δv` | 0,29 m | 5·10⁻⁵ ° |
| + corrections de coning et de sculling à deux échantillons | 8 mm | 3·10⁻⁸ ° |

![Validation du strapdown](docs/img/step2_strapdown_validation.png)

L'incrément d'angle mesuré est l'intégrale de la vitesse angulaire, pas la rotation exacte sur l'intervalle. Les deux diffèrent quand l'axe de rotation bouge (coning), ce qui arrive ici à chaque mise en virage. La compensation de rotation et le sculling corrigent l'équivalent pour la vitesse.

Avec cette dynamique douce et une IMU à 200 Hz, même la variante naïve reste très loin sous l'erreur des capteurs (1 m contre des centaines). Les corrections pèsent davantage avec des vibrations ou une IMU plus lente, que cette simulation ne modélise pas encore. Un test compare les trois variantes et exige que chaque correction réduise l'erreur, ce qui verrouille le signe des termes : un produit vectoriel inversé rendrait la variante complète moins bonne que la précédente.

### IMU MEMS seule, comparée au budget d'erreur

| | Bruit blanc | Biais initial | Marche aléatoire du biais |
|---|---|---|---|
| Gyromètres | 0,01 °/s/√Hz (0,6 °/√h) | 0,02 °/s (72 °/h) | 2·10⁻⁵ rad/s/√s |
| Accéléromètres | 0,02 m/s²/√Hz | 0,05 m/s² (≈ 5 mg) | 5·10⁻⁴ m/s²/√s |

Chaque axe a ses propres tirages. Les valeurs sont celles d'une IMU MEMS grand public non calibrée en vol, avec les vibrations incluses dans le bruit.

En vol rectiligne, en partant de l'état vrai, chaque source d'erreur donne une loi de croissance connue de l'erreur de position horizontale, par axe. Une erreur d'inclinaison δ fait lire à l'accéléromètre `g·δ` à l'horizontale, d'où le facteur `g` et les deux intégrations de plus pour les gyros :

| Source | Erreur de position par axe |
|---|---|
| biais accéléro `b_a` | `b_a t²/2` |
| biais gyro `b_g` | `g b_g t³/6` |
| bruit accéléro `N_a` | `N_a √(t³/3)` |
| bruit gyro `N_g` | `g N_g √(t⁵/20)` |
| dérive du biais accéléro `K_a` | `K_a √(t⁵/20)` |
| dérive du biais gyro `K_g` | `g K_g √(t⁷/252)` |

![Navigation à l'estime contre budget](docs/img/step2_dead_reckoning.png)

Sur 500 vols simulés, l'erreur RMS colle au budget à 2–4 % près de 10 à 120 s, ce qui reste dans l'intervalle de ±6 % attendu pour une RMS estimée sur 500 tirages. Elle atteint 160 m par axe à 60 s et 1,1 km à 120 s. Au-delà de 45 s environ, c'est le biais gyro qui domine : c'est lui que la fusion GNSS devra estimer en premier.

Le cap a un statut à part. Un biais sur le gyro de lacet fait dériver le cap, mais en vol rectiligne non accéléré la force spécifique est verticale, et tourner le cap ne change rien à la position. Un test le vérifie : 0,05 °/s de biais donne 6° d'erreur de cap en 120 s et moins de 50 cm d'erreur de position. Le cap n'est donc pas observable en ligne droite avec une IMU et un GNSS seuls. Il le devient dans les virages et les accélérations, ou avec un magnétomètre.

### Écarts rencontrés

Dans le test du virage stabilisé, j'attendais une force latérale nulle, puisque le virage est coordonné. Le test a mesuré 6,7 mm/s². La loi `dψ/dt = g tan φ / V` suppose une portance perpendiculaire à la trajectoire, mais avec 3° d'assiette l'axe latéral de l'avion n'est plus exactement horizontal. Il reste alors `g sin φ (1 − cos θ)`, soit 6,7 mm/s² à 30° d'inclinaison. C'est sans conséquence, parce que la vérité et l'IMU restent exactement cohérentes entre elles. Le test vérifie maintenant cette valeur exacte au lieu de zéro.

## Provenance des résultats

Chaque `docs/stepN_results.md` se termine par une section « Provenance », complétée par un fichier JSON voisin. Le script calcule lui-même, au moment d'écrire ses résultats, une empreinte SHA-256 des modules qu'il a réellement importés et de son propre code. Il enregistre aussi les paramètres, la graine, les versions des bibliothèques et le commit git quand il est disponible. Les fins de ligne sont normalisées avant le hachage, donc un clone Windows et un clone Linux du même commit donnent la même empreinte.

`scripts/check_provenance.py` compare ces empreintes au code actuel et liste les fichiers modifiés depuis. Le test `test_committed_results_match_current_code` fait la même chose en CI : modifier un module sans relancer le script qui produit les chiffres fait échouer les tests.

## Organisation

```
src/navsim/
  trajectory.py       étape 1 : profil vertical (vérité terrain)
  sensors.py          étape 1 : accéléromètre vertical et baromètre
  kf_altitude.py      étape 1 : filtre de Kalman, discrétisation Van Loan
  runner.py           étape 1 : boucle de simulation, NEES/NIS, navigation à l'estime
  rotations.py        quaternions et angles d'Euler, conventions NED/FRD
  trajectory3d.py     étape 2 : trajectoire 3D et incréments IMU idéaux
  imu.py              étape 2 : modèle d'erreur IMU 6 axes
  strapdown.py        étape 2 : intégration strapdown, budget d'erreur analytique
  provenance.py       empreinte du code et des paramètres de chaque résultat
scripts/
  step1_altitude.py   figures et tableau de l'étape 1
  step2_strapdown.py  figures et tableau de l'étape 2
  check_provenance.py vérifie que les résultats de docs/ correspondent au code
tests/
  test_step1.py       vérité cohérente, Van Loan, covariance définie positive, cohérence NEES
  test_step2.py       conventions, cohérence IMU/vérité, strapdown, Monte-Carlo contre budget
  test_provenance.py  empreinte stable, résultats à jour
docs/
  stepN_results.md    tableaux générés par les scripts, avec leur provenance
  stepN_provenance.json
  img/                figures
```

## Suite

3. EKF à état d'erreur à 15 états (position, vitesse, attitude en quaternion, biais gyro, biais accéléro) avec le GNSS.
4. GNSS retardé (100 à 200 ms) fusionné à l'horizon retardé avec un buffer d'état.
5. Baro, magnétomètre, Pitot et estimation du vent.
6. Modes dégradés : perte GNSS, perturbation magnétique, vibrations. Gating des innovations et détection de capteur défaillant.
7. Rejeu de logs de vol PX4 réels et comparaison avec l'EKF2 embarqué.
8. Portage C++ (matrices de taille fixe, sans allocation dynamique) et comparaison avec la référence Python sur les mêmes logs.
