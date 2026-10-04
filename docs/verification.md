# Vérification

Une bonne partie du code de ce dépôt a été écrite avec un assistant IA (Claude), et les tests aussi. Un test écrit en même temps que le code qu'il vérifie risque de partager ses erreurs : la même convention de rotation, le même signe, la même compréhension fausse d'une densité de bruit. La vérification est donc organisée en couches, et chaque couche passe par un chemin différent de celui du code.

`python -m pytest` lance tout sauf les tests de mutation (environ onze minutes).

## 1. Tests de chaque étape

`tests/test_step1.py` à `tests/test_step8.py`. Ils vérifient ce que chaque étape affirme : cohérence de la vérité terrain, consistance des filtres (NEES, NIS), effet de chaque correction du strapdown, observabilité du cap, comportement face à la latence, estimation du vent et apport du baromètre (à tirages identiques, avec et sans), rejet des pannes par le test d'innovation, et les deux blocages de l'étape 6 avec leurs protections : chaque blocage est reproduit sans protection, puis corrigé avec.

## 2. Oracles indépendants

`tests/test_oracles.py`. Chaque test compare le code du projet à quelque chose avec lequel il ne partage aucune ligne.

| Ce qui est vérifié | Comparé à |
|---|---|
| Quaternions, angles d'Euler, matrices de rotation, produit, exponentielle | `scipy.spatial.transform.Rotation` |
| Discrétisation de Van Loan (Φ, Qd, Γ) de l'étape 1 | intégration de l'équation de Lyapunov `dP/dt = AP + PAᵀ + LQLᵀ` par `solve_ivp`, sur un modèle retapé depuis la docstring |
| Filtre de Kalman de l'étape 1 | moindres carrés pondérés en bloc : sans bruit de processus, les deux doivent donner exactement la même estimation et la même covariance |
| Distribution du NEES de l'étape 1 | loi du χ² à 2 degrés de liberté (test de Kolmogorov-Smirnov sur 300 runs), pas seulement sa moyenne |
| Générateur de trajectoire et strapdown | intégration des équations continues par `solve_ivp` (DOP853, tolérance 10⁻¹¹) sur 60 s avec deux boucles |
| Vitesses angulaires et accélérations analytiques | différences finies sur l'attitude et la vitesse générées |
| Budget d'erreur de l'étape 2 | formules fermées sur des cas déterministes à une seule source d'erreur (biais accéléro, biais gyro) |
| Matrice de transition de l'EKF | jacobienne numérique colonne par colonne (différences centrées) du strapdown non linéaire, en ligne droite et en virage |
| Mise à jour GNSS en forme de Joseph | forme information `(P⁻¹ + HᵀR⁻¹H)⁻¹` |
| Covariance propagée par l'EKF | covariance empirique de 600 runs en boucle ouverte, avec toutes les erreurs, puis avec le bruit blanc seul |
| Générateur IMU en flux | sa propre spécification : écart-type par échantillon `N√dt`, biais initial, marche aléatoire `K√t` |
| Unités physiques du bruit de processus | même vol propagé à 100 Hz et à 200 Hz : même covariance |
| Gestion de la latence | géométrie déterministe : à 17 m/s avec 150 ms de retard ignoré, l'estimation doit être 2,55 m en arrière ; compensée ou à horizon retardé, à zéro |
| Jacobiennes des mesures baro, magnétomètre, Pitot et dérapage (étape 5) | dérivée numérique de fonctions de mesure réécrites depuis leur définition physique avec scipy, autour d'états tirés au hasard ; le test capture le `H` et l'innovation que le filtre utilise vraiment |
| Cap magnétique compensé en inclinaison | champ terrestre tourné par scipy pour 200 attitudes aléatoires |
| Covariance initiale du vent (premier échantillon Pitot) | Monte-Carlo de 40 000 tirages de la relation non linéaire, termes croisés vent/cap/vitesse compris |
| Générateurs magnétomètre et baro | leur spécification : biais initial, bruit, marche aléatoire, stationnarité de la dérive |
| Rejeu (étape 7) | des logs simulés avec les défauts d'un vrai log (intervalles IMU de 5 ou 10 ms au hasard, autopilote monté tourné de 8°, vitesse air fausse de 20 %) : la vérité y est connue, et les états de calibration doivent retrouver le décalage et l'échelle |
| Mesure de cap, états d'échelle de vitesse air et de décalage de dérapage (étape 7) | dérivée numérique de fonctions de mesure réécrites avec scipy, à plusieurs tangages |
| Rediscrétisation pour un intervalle IMU quelconque (étape 7) | un filtre construit directement à cet intervalle |
| Projection locale des positions GNSS (étape 7) | conversion exacte par coordonnées cartésiennes sur la même sphère |
| Lecteur de logs PX4 (étape 7) | un petit log réel du dépôt pyulog : incréments IMU recalculés depuis les champs bruts, retard GNSS, champ magnétique attendu à l'endroit du vol |
| Portage C++ (étape 8) | la référence Python, état par état à chaque époque GNSS, NIS et décisions du test d'innovation comprises, sur des vols simulés qui passent par toutes les branches de la boucle |
| Absence d'allocation dynamique en C++ (étape 8) | la garde d'Eigen (`EIGEN_RUNTIME_NO_MALLOC`) et un `operator new` global qui compte les allocations, lui-même vérifié par un test |
| Test d'innovation (étape 6) | 20 000 innovations tirées dans la covariance que le filtre prédit : le taux de rejet doit suivre la probabilité choisie (test binomial). Cela vérifie ensemble le NIS, le seuil et le nombre de degrés de liberté |

## 3. Propriétés

`tests/test_properties.py`, avec [hypothesis](https://hypothesis.readthedocs.io). Au lieu de quelques entrées choisies à la main, hypothesis en tire des centaines et réduit tout échec au contre-exemple le plus simple. Les invariants vérifiés : une rotation conserve les longueurs, le produit de quaternions compose les rotations, `exp` et `log` sont inverses, une covariance reste symétrique définie positive après n'importe quelle mise à jour, et une mesure ne peut jamais ajouter d'incertitude.

Ce test a trouvé un vrai défaut dès sa première exécution. Avec un écart-type de dérive baro positif mais minuscule (2·10⁻¹⁶⁴ m), le filtre de l'étape 1 activait son quatrième état, mais la variance de cet état valait zéro en virgule flottante, et P devenait singulière. Je n'aurais jamais écrit ce cas à la main. Le filtre refuse maintenant une valeur entre 0 et 10⁻⁶ m, et un test de non-régression garde le cas. L'EKF fait la même chose pour les états de biais GNSS.

## 4. Tests métamorphiques

Dans le même fichier. Ils n'ont pas besoin de réponse de référence, seulement d'une relation entre deux exécutions :

- le même vol fait avec un cap initial de 0° puis de 90°, avec exactement les mêmes erreurs IMU, doit donner les mêmes erreurs de navigation tournées de 90°. Cela vérifie d'un coup les conventions de repère du générateur et du strapdown ;
- doubler les biais doit doubler l'erreur qu'ils causent (régime linéaire) ;
- dans un vent constant, l'avion vole la même trajectoire par rapport à l'air : attitude, vitesses angulaires et force spécifique identiques à celles du vol sans vent, position décalée de `w·t` (invariance galiléenne, `tests/test_step5.py`).

## 5. Tests de mutation

`scripts/mutation_check.py` introduit 55 bugs plausibles, un par un, dans une copie du dépôt : un signe dans la matrice de transition, une correction d'attitude injectée dans le mauvais repère, une densité de bruit appliquée en `dt` au lieu de `√dt`, la gravité de signe inversé, un échantillon IMU de décalage... Il lance ensuite la suite de tests sur chaque copie. Une suite de tests qu'on n'a jamais vue échouer sur un bug connu ne prouve pas grand-chose ; c'est le test des tests. Le rapport est dans [`mutation_report.md`](mutation_report.md).

La première exécution a montré deux faiblesses réelles :

- **M16, densité de bruit gyro mal appliquée dans le générateur IMU en flux :** aucun test ne l'a vue. Ce générateur, utilisé par tous les Monte-Carlo des étapes 3 et 4, n'avait aucun test direct, et l'erreur de bruit gyro était noyée sous les biais dans les tests de cohérence.
- **M03, bruit de processus gyro en `dt²` :** il n'a été détecté que par hasard, par un test de l'étape 4 qui ne le visait pas, après 5 minutes de suite complète.

J'ai ajouté un test qui confronte le générateur à sa spécification, et une variante « bruit seul » de la comparaison covariance/Monte-Carlo. Les deux mutations sont maintenant détectées directement.

Une mutation survit, et c'est attendu : **M06**, la forme de Joseph remplacée par la forme courte. C'est un mutant équivalent : avec le gain optimal, les deux formes sont égales en arithmétique exacte.

### M05 : la mutation qui avait raison

M05 inverse le signe de la jacobienne de réinitialisation de l'EKF, la matrice qui fait tourner la covariance d'attitude quand on injecte la correction dans le quaternion. Jusqu'à l'étape 4, elle survivait, et je l'avais classée « attendue » : un effet du second ordre, sous les tolérances.

À l'étape 5, un nouveau test l'a détectée : celui qui vérifie que le filtre devient trop confiant quand il apprend le biais du magnétomètre en ligne droite. Avec la mutation, le NEES d'attitude passait de 2,6 à 1,2. Une mutation qui rend le filtre plus cohérent est suspecte, donc j'ai dérivé la jacobienne numériquement avec scipy. C'était le code d'origine qui était faux, pas la mutation. J'avais pris `I − [δθ/2 ×]`, la formule d'une erreur d'attitude locale (`q_vrai = q ⊗ exp(δθ)`). Ce filtre utilise une erreur globale (`q_vrai = exp(δθ) ⊗ q`), et il faut alors `I + [δθ/2 ×]`. Écart à la dérivée numérique : 5·10⁻⁴ avec le bon signe, 5·10⁻² avec l'ancien.

Aucun test ne le voyait, parce que l'effet est petit tant que les corrections d'attitude sont petites. Il devient visible quand le filtre accumule beaucoup de petites corrections sur une direction mal observée, ce qui est exactement le cas du cap en ligne droite. La correction a changé quelques résultats de l'étape 3 : le NEES d'attitude avant le premier virage passe de 1,45 à 1,20. Le README raconte l'écart que j'avais attribué à tort à la seule linéarisation.

Ce que j'en retiens. Classer une mutation « attendue » était une hypothèse que j'aurais dû vérifier tout de suite avec un oracle. La formule `I − [δθ/2 ×]` est juste pour une erreur locale et fausse pour l'erreur globale de ce filtre. Et c'est un test qui vérifie un comportement connu pour être mauvais (E2b, le biais appris en ligne droite) qui a trouvé le bug.

`tests/test_oracles.py::test_reset_jacobian_matches_rotation_composition` compare maintenant la jacobienne à la dérivée numérique de la composition exacte, et M05 est détectée.

Les mutations M26 à M34 visent l'étape 5 : signes des jacobiennes magnétomètre, Pitot et baro, signe de la corrélation vent/cap, apprentissage du biais magnétomètre inversé (en ligne droite au lieu des virages), accélération du vent oubliée dans la force spécifique. M28 reproduit une erreur que j'ai réellement faite en écrivant la mesure de dérapage : la dérivée de `1/V` oubliée dans la jacobienne. Je l'avais trouvée avec une dérivée numérique ; la mutation vérifie que les tests la trouvent seuls.

M35 à M41 visent l'étape 6 : test d'innovation inversé, mesure rejetée mais fusionnée quand même, mauvais nombre de degrés de liberté, protection contre le blocage désactivée, pannes simulées ignorées. À la première exécution, M37 a survécu : un seuil calculé sur 3 degrés de liberté au lieu de 6 rejette environ 1 % des bonnes positions GNSS au lieu de 0,1 %, et aucun test ne regardait le taux de rejet en vol. Un test le vérifie maintenant avant la perte GNSS (test binomial), et M37 est détectée.

M42 à M48 visent l'étape 7 : intervalle IMU nominal au lieu du vrai, couplage cap / inclinaison oublié, signes des états de calibration, retard GNSS appliqué à l'envers, précision du récepteur mal répartie, cos(latitude) oublié dans la projection.

M49 à M55 visent le C++ : signes de la matrice de transition, de la jacobienne de réinitialisation, du décalage de dérapage et de la corrélation vent / cap, test d'innovation inversé, couplage cap / inclinaison oublié, protection du Pitot oubliée dans la boucle. Le test les détecte parce qu'il compare le C++ à la version Python, elle-même vérifiée par tout le reste. Chaque mutation C++ demande une recompilation complète, dans un dossier neuf.

Le script s'arrête en erreur si une mutation survit sans être marquée comme attendue. En CI, il tourne chaque semaine et à la demande (environ une heure).

## 6. Documentation et résultats

- **Provenance :** chaque fichier de résultats porte l'empreinte du code qui l'a produit. `scripts/check_provenance.py` et un test signalent tout résultat périmé, y compris le rapport de mutation, dont l'empreinte couvre les sources et les tests.
- **Chiffres du README :** les tableaux de résultats du README sont marqués avec leur fichier source (`<!-- source: docs/stepN_results.md -->`). `tests/test_docs.py` vérifie que chaque nombre de ces tableaux se retrouve dans le fichier généré, à l'arrondi affiché près. Changer 0,17 en 0,71 dans le README fait échouer la suite.
- **Images et arborescence :** les images citées et les fichiers listés dans l'arborescence doivent exister.

## 7. Analyse statique et CI

ruff tourne avec des règles orientées bugs : noms non définis ou inutilisés, `zip` sans `strict=`, variables de boucle inutilisées. Il a trouvé un cas limite réel. La fonction qui repère les virages pour les figures perdait silencieusement un virage encore en cours à la fin du vol, parce que `zip` tronquait la liste des débuts. C'est corrigé et vérifié.

La CI GitHub lance ruff, puis toute la suite avec la couverture de code, puis la vérification de provenance, à chaque push.

## Ce que ces tests ne couvrent pas

- **Le monde réel.** Dans les tests, la vérité terrain et le filtre partagent les mêmes hypothèses : Terre plate, pas de rotation terrestre, vent horizontal sans turbulence, erreurs capteurs gaussiennes. L'étape 7 rejoue de vrais vols, mais sans vérité terrain : le filtre y est jugé contre l'EKF2 et contre le GNSS après des pertes simulées. Ces vols ont déjà montré trois hypothèses fausses (magnétomètre, dérapage, vitesse air). Les tests de l'étape 7 reproduisent ces défauts en simulation, mais seulement ceux-là.
- **Les tolérances.** Elles sont choisies à la main. Une tolérance trop large laisse passer un bug ; les tests de mutation sont là pour le mesurer, mais seulement sur les bugs qu'on a pensé à y mettre.
- **Les tests statistiques.** Ils utilisent des graines fixes, donc ils sont reproductibles. Avec une autre graine, un test cohérent peut échouer rarement, par construction (un intervalle à 95 % est dépassé une fois sur vingt).
- **La relecture.** Ces couches réduisent le risque qu'un bug passe, elles ne remplacent pas la relecture par quelqu'un qui comprend les équations.
