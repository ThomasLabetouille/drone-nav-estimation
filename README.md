# drone-nav-estimation

[![tests](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml/badge.svg)](https://github.com/ThomasLabetouille/drone-nav-estimation/actions/workflows/tests.yml)

Simulation de capteurs et estimation d'état pour un drone à voilure fixe, en Python, avec un portage C++ prévu une fois les algorithmes validés.

Le projet avance par étapes. Chacune ajoute des capteurs ou des états au filtre et se valide avant de passer à la suivante. Quatre étapes sont faites :

1. la voie verticale : un filtre de Kalman qui fusionne un accéléromètre et un baromètre ;
2. la navigation inertielle en 3D : trajectoire de voilure fixe, IMU 6 axes, intégration strapdown, et l'erreur d'une IMU MEMS seule comparée à son budget analytique ;
3. un EKF à état d'erreur à 15 états qui fusionne l'IMU et le GNSS, avec ce qu'il apprend à chaque manœuvre et ce qu'il ne peut pas apprendre en ligne droite ;
4. un GNSS réaliste : erreurs corrélées dans le temps, estimées par trois états de plus, et 150 ms de latence gérée par un filtre à horizon retardé.

![Estimation d'altitude sur une mission complète](docs/img/step1_estimation.png)

## Lancer

```bash
pip install -e ".[dev]"
python -m pytest                       # environ quatre minutes
python scripts/step1_altitude.py       # étape 1, ~45 s
python scripts/step2_strapdown.py      # étape 2, ~50 s
python scripts/step3_eskf.py           # étape 3, ~2 min 30
python scripts/step4_gnss.py           # étape 4, ~5 min
python scripts/step2_strapdown.py --runs 100   # plus rapide, moins de Monte-Carlo
python scripts/check_provenance.py     # les résultats de docs/ correspondent-ils au code actuel ?
python scripts/mutation_check.py       # les tests détectent-ils des bugs connus ? ~15 min
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

<!-- source: docs/step1_results.md -->
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

<!-- source: docs/step2_results.md -->
| Variante | Position après 8 min | Attitude, erreur max |
|---|---|---|
| naïve : `v += C(q) Δv` | 1,14 m | 5·10⁻⁵ ° |
| + compensation de rotation `½ Δθ × Δv` | 0,29 m | 5·10⁻⁵ ° |
| + corrections de coning et de sculling à deux échantillons | 0,0084 m | 3·10⁻⁸ ° |

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

## Étape 3 : EKF à état d'erreur, IMU + GNSS

![Ce que le filtre apprend, et quand](docs/img/step3_observability.png)

### Filtre

L'état nominal (position, vitesse, quaternion d'attitude, biais gyro, biais accéléro) est propagé par le strapdown de l'étape 2, avec des incréments corrigés des biais estimés. Le filtre estime l'erreur de cet état nominal, définie comme vérité moins nominal, sur 15 états :

```
δx = [δp, δv, δθ, δb_g, δb_a]          q_vrai = exp(δθ) ⊗ q_nominal,  δθ en NED

d(δp)/dt = δv
d(δv)/dt = −[f_n ×] δθ − C δb_a + C n_a
d(δθ)/dt = −C δb_g + C n_g
d(δb)/dt = marche aléatoire
```

Le terme `−[f_n ×] δθ` rend l'attitude observable à partir de la position et de la vitesse : une erreur d'attitude fait projeter une partie de la force spécifique sur le mauvais axe. En vol rectiligne non accéléré, `f_n` est verticale, et une erreur de cap (une rotation autour de la verticale) ne produit rien. C'est le constat de l'étape 2, retrouvé ici dans la covariance.

- La matrice de transition est développée à l'ordre 2, `Φ = I + F dt + (F dt)²/2`, et écrite bloc par bloc parce que `F` est creuse.
- Le bruit de processus discret est constant et diagonal : `C (N² I) Cᵀ = N² I`, puisque `C` est une rotation.
- La mise à jour GNSS (position et vitesse) se fait en forme de Joseph.
- L'erreur estimée est ensuite injectée dans l'état nominal, puis remise à zéro, avec la jacobienne de réinitialisation `I − [δθ/2 ×]` sur le bloc d'attitude.
- Tout est vectorisé sur les runs : un Monte-Carlo de 100 filtres tourne en une seule passe.

Le GNSS fournit position et vitesse à 5 Hz (σ = 1,5 m à l'horizontale, 3 m en vertical, 0,1 m/s), sans latence et avec des erreurs blanches. L'initialisation se fait en vol, sur le premier point GNSS. Le roulis et le tangage sont connus à 2° près (nivellement sur l'accéléromètre), le cap à 5° près (route GNSS, sans vent).

Le code est dans `src/navsim/eskf.py`, `src/navsim/fusion.py` et `src/navsim/gnss.py`.

Un test vérifie `F` directement. Il perturbe l'état vrai d'une petite erreur sur les 15 états, propage la vérité et l'état nominal dans le strapdown non linéaire pendant 0,2 s, et compare l'erreur obtenue au produit des `Φ` du filtre. Il le fait une fois en ligne droite et une fois en virage, et l'écart doit rester sous 5 %. Un signe faux dans un bloc le fait échouer immédiatement.

### Résultats (100 runs, mission de l'étape 2)

<!-- source: docs/step3_results.md -->
| Après convergence (t > 120 s) | Nord | Est | Bas |
|---|---|---|---|
| Position, RMS [m] | 0,17 | 0,17 | 0,34 |
| Vitesse, RMS [m/s] | 0,034 | 0,034 | 0,044 |

L'inclinaison est connue à 0,064° RMS et le cap à 0,43° RMS.

La précision en position est optimiste. Les erreurs GNSS simulées sont blanches, donc le filtre peut les moyenner à volonté. Les erreurs d'un vrai récepteur sont corrélées sur des dizaines de secondes, et ce moyennage ne marche plus. Elles arrivent avec la latence, dans une étape suivante.

### Ce que le filtre apprend, et quand

Les zones grises de la figure du haut sont les virages. Le premier commence à 71 s.

<!-- source: docs/step3_results.md -->
| Incertitude moyenne (1σ) | Avant le premier virage (65 s) | Après (90 s) |
|---|---|---|
| Cap | 4,0° | 0,29° |
| Inclinaison | 0,29° | 0,063° |
| Biais accéléro x (avant) | 0,039 m/s² | 0,008 m/s² |
| Biais accéléro z | 0,006 m/s² | 0,0035 m/s² |

- **Cap :** il reste à 4° tant que l'avion va tout droit, puis tombe à 0,3° dès le premier virage. Il remonte ensuite à chaque ligne droite (jusqu'à 1° avant la dernière boucle), parce que le biais du gyro de lacet est lui-même mal connu.
- **Inclinaison et biais accéléro horizontal :** en vol stabilisé, une erreur d'inclinaison δ et un biais horizontal `b` produisent la même erreur d'accélération (`g·δ` contre `b`). Le filtre ne voit que leur somme. L'inclinaison plafonne donc à 0,29° et le biais reste presque à sa valeur initiale. Le virage change l'orientation de la force spécifique par rapport à l'avion, ce qui sépare les deux.
- **Biais accéléro vertical :** il est appris en une vingtaine de secondes, puisque la verticale est mesurée directement.

![Biais gyro](docs/img/step3_gyro_bias.png)

Le biais du gyro de lacet ne bouge pas avant les virages. Il passe de 0,020 à 0,007 °/s pendant les deux boucles, puis se dégrade à nouveau en ligne droite, sous l'effet de sa marche aléatoire.

![Un vol](docs/img/step3_single_run.png)

### Cohérence

<!-- source: docs/step3_results.md -->
| Bloc | NEES ou NIS moyen / ddl (attendu : 1) | Temps dans l'intervalle à 95 % |
|---|---|---|
| Position | 1,00 | 98 % |
| Vitesse | 1,01 | 94 % |
| Attitude | 1,05 | 81 % |
| Biais gyro | 1,02 | 86 % |
| Biais accéléro | 1,04 | 96 % |
| NIS GNSS | 1,00 | 95 % |

![Cohérence](docs/img/step3_consistency.png)

- **Position, vitesse et NIS** sont cohérents.
- **Biais :** leur moyenne est bonne, mais leur temps dans l'intervalle varie d'une graine à l'autre. Avec deux autres graines, j'obtiens 91 et 92 % pour les biais gyro, 90 et 94 % pour les biais accéléro. Une erreur de biais évolue lentement, donc la moyenne sur 100 runs fait de longues excursions au lieu de fluctuer autour de 1. La fraction de temps hors de la bande est alors beaucoup plus dispersée que pour la position, qui se décorrèle en quelques secondes.
- **Attitude :** le défaut est réel et localisé. Il fait l'objet de la section suivante.

### Écarts rencontrés

Avant le premier virage, le NEES d'attitude vaut 1,45 : le filtre est trop sûr de lui. Après 100 s, il revient à 1. Mon hypothèse était la linéarisation. Avec 5° d'erreur de cap, les termes du second ordre que l'EKF néglige (le produit de l'erreur de cap par l'erreur d'inclinaison dans `f × δθ`) sont du même ordre que ce que le filtre estime en ligne droite. Pour le vérifier, j'ai relancé le même vol en ne changeant que l'erreur de cap initiale :

<!-- source: docs/step3_results.md -->
| σ du cap initial | NEES d'attitude / ddl, de 5 s au premier virage | Après 100 s |
|---|---|---|
| 5° | 1,45 | 0,98 |
| 2° | 1,12 | 1,02 |
| 1° | 1,04 | 1,01 |

L'excès disparaît quand l'erreur initiale diminue, ce qui confirme l'hypothèse. Trois remèdes sont possibles et aucun n'est encore implémenté :

- un meilleur cap initial (magnétomètre, étape 5) ;
- gonfler la covariance d'attitude jusqu'au premier virage ;
- une mise à jour itérée ou sans parfum (UKF).

Le script `step3_eskf.py` refait cette comparaison à chaque exécution.

## Étape 4 : GNSS réaliste, erreurs corrélées et latence

![Latence GNSS](docs/img/step4_latency.png)

Le GNSS de l'étape 3 avait des erreurs blanches et aucune latence, deux hypothèses qu'aucun récepteur ne respecte. Ici, l'erreur totale de position reste la même (1,5 m à l'horizontale, 3 m en vertical), mais elle est découpée autrement :

- un bruit blanc de 0,5 m / 1 m ;
- une erreur de Gauss-Markov du premier ordre de 1,4 m / 2,8 m, corrélée sur 60 s (atmosphère, trajets multiples) ;
- une vitesse Doppler qui reste blanche ;
- une latence de 150 ms : la mesure décrit l'état à `t` mais arrive au filtre à `t + 150 ms`.

Toutes les configurations tournent sur la même mission, avec 50 runs et la même graine. Les erreurs sont celles de l'état délivré en temps réel, celui qu'utiliserait le pilote automatique, et sont mesurées après 120 s.

### Erreurs corrélées

<!-- source: docs/step4_results.md -->
| | Position horiz. | NEES position | NIS GNSS |
|---|---|---|---|
| A : erreurs blanches (étape 3) | 0,24 m | 1,02 | 1,00 |
| B : erreurs corrélées, traitées comme blanches | 1,78 m | 53 | 0,65 |
| C : erreurs corrélées, estimées par 3 états de plus | 1,18 m | 0,93 | 1,00 |

![Erreurs corrélées](docs/img/step4_correlated.png)

**B.** Le filtre traite une erreur qui dure une minute comme si elle changeait à chaque mesure. Il croit donc pouvoir la moyenner et annonce ±0,5 m (3σ), alors qu'il suit l'erreur du récepteur jusqu'à 2,5 m. C'est la même situation que la dérive du baro de l'étape 1.

Le NIS, seul test disponible en vol, dit même le contraire de la réalité : il vaut 0,65, donc le filtre a l'air trop prudent. La raison est que deux mesures successives portent presque la même erreur. L'innovation ne voit plus que la partie blanche (0,5 m), alors que le filtre attend 1,5 m.

**C.** Trois états de biais GNSS (Gauss-Markov, τ = 60 s) passent le filtre à 18 états. À l'initialisation sur le premier point, l'erreur de position vaut `−(b + n)` et l'erreur de biais vaut `b`, d'où un terme croisé négatif dans `P0`, exactement comme pour le baro de l'étape 1. Le filtre redevient cohérent, et l'erreur baisse d'un tiers (1,18 m contre 1,78 m). La vitesse Doppler n'est pas biaisée : intégrée, elle donne une position relative fiable qui permet de séparer une partie du biais de la vraie position.

### Latence de 150 ms

<!-- source: docs/step4_results.md -->
| | Position horiz. | Cap | NEES attitude | NIS GNSS |
|---|---|---|---|---|
| latence ignorée | 2,97 m | 2,63° | 61 | 1,18 |
| latence compensée sur la position (`z += v · 150 ms`) | 2,34 m | 2,57° | 58 | 1,25 |
| horizon retardé + prédicteur de sortie | 1,23 m | 0,46° | 1,02 | 1,00 |

**Latence ignorée.** En ligne droite, le filtre place l'avion là où il était 150 ms plus tôt, soit 2,5 m de retard à 17 m/s (3,3 m pendant la pointe à 22 m/s, bien visible sur la figure). En virage, c'est pire, parce que la mesure se trompe aussi de direction de vitesse. À 19 °/s de taux de virage, 150 ms font 2,9° : un écart latéral de 0,86 m/s que le filtre ne peut expliquer que par une erreur d'attitude ou de biais gyro. Le cap et le biais du gyro de lacet sont faussés à chaque virage, puis le cap dérive à chaque ligne droite (la rampe entre 300 et 400 s). Le NEES d'attitude monte à 61.

**Compensation de la position.** Avancer la position mesurée de `v · 150 ms` corrige le retard en ligne droite. La vitesse mesurée reste en retard en virage, donc l'attitude reste fausse et l'erreur de position revient dès le premier virage.

**Horizon retardé.** Le filtre tourne 150 ms dans le passé. Chaque mesure GNSS y arrive exactement à sa date, et le filtre lui-même est celui de C. L'état présent est obtenu en réintégrant les 30 derniers échantillons IMU gardés en mémoire (le prédicteur de sortie), sans covariance. C'est le principe de l'EKF2 de PX4. Toutes les erreurs reviennent au niveau de C (1,23 m contre 1,18 m), et la cohérence aussi.

Le NIS de la latence ignorée ne vaut que 1,18 en moyenne. En ligne droite, il reste à 1 : le filtre absorbe le retard en décalant son état, et rien ne le signale. Il ne monte qu'en virage (jusqu'à 3,7 pendant les virages en S). Un NIS qui ne grimpe qu'en virage est la signature à chercher dans des logs réels.

Le code est dans `src/navsim/fusion.py` (les trois stratégies), `src/navsim/eskf.py` (états de biais GNSS) et `src/navsim/gnss.py`.

### Écarts rencontrés

La généralisation du filtre (15 ou 18 états, matrice `H` explicite, boucle avec file d'échantillons IMU) touchait du code dont dépend l'étape 3. J'ai relancé l'étape 3 et comparé son tableau de résultats avec l'ancien, en ignorant la section provenance : aucun chiffre n'a changé. Un test vérifie aussi que le mode « horizon retardé » sans latence donne exactement le même filtre que le mode normal.

## Vérification

Une bonne partie du code et des tests de ce dépôt a été écrite avec un assistant IA. Un test écrit en même temps que le code risque de partager ses erreurs, donc la vérification passe aussi par d'autres chemins. Le détail est dans [`docs/verification.md`](docs/verification.md).

- **Oracles indépendants** (`tests/test_oracles.py`). Le code est comparé à ce qu'il ne partage pas :
  - scipy pour les rotations ;
  - un solveur d'EDO pour la trajectoire et le strapdown ;
  - les moindres carrés en bloc pour le Kalman linéaire ;
  - une jacobienne numérique colonne par colonne pour la matrice de transition de l'EKF ;
  - un Monte-Carlo pour la covariance propagée ;
  - des formules fermées pour le budget d'erreur et la latence.
- **Propriétés** (`tests/test_properties.py`, hypothesis). Des invariants sont tirés sur des centaines d'entrées aléatoires. Ce test a trouvé une covariance singulière dans le filtre de l'étape 1, pour une dérive baro positive mais minuscule.
- **Tests métamorphiques.** Le même vol fait avec un cap initial tourné de 90° doit donner les mêmes erreurs, tournées de 90°.
- **Tests de mutation** (`scripts/mutation_check.py`). Vingt-cinq bugs plausibles sont injectés un par un dans une copie du dépôt, et la suite de tests doit les détecter. La première exécution a révélé deux faiblesses, corrigées depuis. Le rapport est dans [`docs/mutation_report.md`](docs/mutation_report.md).
- **Chiffres du README.** Chaque tableau de résultats de ce fichier est vérifié contre le fichier généré correspondant (`tests/test_docs.py`).
- **CI.** ruff (règles orientées bugs), la suite complète avec la couverture de code, et la provenance à chaque push ; les mutations chaque semaine.

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
  gnss.py             étapes 3-4 : récepteur GNSS, erreurs corrélées, latence
  eskf.py             étapes 3-4 : EKF à état d'erreur, 15 ou 18 états
  fusion.py           étapes 3-4 : boucle IMU + GNSS en Monte-Carlo, stratégies de latence
  provenance.py       empreinte du code et des paramètres de chaque résultat
scripts/
  step1_altitude.py   figures et tableau de l'étape 1
  step2_strapdown.py  figures et tableau de l'étape 2
  step3_eskf.py       figures et tableau de l'étape 3
  step4_gnss.py       figures et tableau de l'étape 4
  check_provenance.py vérifie que les résultats de docs/ correspondent au code
  mutation_check.py   injecte des bugs connus et vérifie que les tests les détectent
tests/
  test_step1.py       vérité cohérente, Van Loan, covariance définie positive, cohérence NEES
  test_step2.py       conventions, cohérence IMU/vérité, strapdown, Monte-Carlo contre budget
  test_step3.py       jacobienne F contre propagation non linéaire, cohérence, observabilité du cap
  test_step4.py       erreur GNSS corrélée, états de biais, horizon retardé contre latence ignorée
  test_oracles.py     comparaisons à des références indépendantes (scipy, EDO, moindres carrés...)
  test_properties.py  invariants (hypothesis) et tests métamorphiques
  test_docs.py        chiffres du README contre les résultats générés
  test_provenance.py  empreinte stable, résultats à jour
docs/
  stepN_results.md    tableaux générés par les scripts, avec leur provenance
  stepN_provenance.json
  verification.md     stratégie de vérification
  mutation_report.md  résultat des tests de mutation
  img/                figures
```

## Suite

5. Baro, magnétomètre, Pitot et estimation du vent.
6. Modes dégradés : perte GNSS, perturbation magnétique, vibrations. Gating des innovations et détection de capteur défaillant.
7. Rejeu de logs de vol PX4 réels et comparaison avec l'EKF2 embarqué.
8. Portage C++ (matrices de taille fixe, sans allocation dynamique) et comparaison avec la référence Python sur les mêmes logs.
