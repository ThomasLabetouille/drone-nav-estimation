# Vérification

Une bonne partie du code de ce dépôt a été écrite avec un assistant IA (Claude), et les tests aussi. Un test écrit en même temps que le code qu'il vérifie risque de partager ses erreurs : la même convention de rotation, le même signe, la même compréhension fausse d'une densité de bruit. La vérification est donc organisée en couches, et chaque couche passe par un chemin différent de celui du code.

`python -m pytest` lance tout sauf les tests de mutation (environ quatre minutes).

## 1. Tests de chaque étape

`tests/test_step1.py` à `tests/test_step4.py`. Ils vérifient ce que chaque étape affirme : cohérence de la vérité terrain, consistance des filtres (NEES, NIS), effet de chaque correction du strapdown, observabilité du cap, comportement face à la latence.

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

## 3. Propriétés

`tests/test_properties.py`, avec [hypothesis](https://hypothesis.readthedocs.io). Au lieu de quelques entrées choisies à la main, hypothesis en tire des centaines et réduit tout échec au contre-exemple le plus simple. Les invariants vérifiés : une rotation conserve les longueurs, le produit de quaternions compose les rotations, `exp` et `log` sont inverses, une covariance reste symétrique définie positive après n'importe quelle mise à jour, et une mesure ne peut jamais ajouter d'incertitude.

Ce test a trouvé un vrai défaut dès sa première exécution. Avec un écart-type de dérive baro positif mais minuscule (2·10⁻¹⁶⁴ m), le filtre de l'étape 1 activait son quatrième état, mais la variance de cet état valait zéro en virgule flottante, et P devenait singulière. Je n'aurais jamais écrit ce cas à la main. Le filtre refuse maintenant une valeur entre 0 et 10⁻⁶ m, et un test de non-régression garde le cas. L'EKF fait la même chose pour les états de biais GNSS.

## 4. Tests métamorphiques

Dans le même fichier. Ils n'ont pas besoin de réponse de référence, seulement d'une relation entre deux exécutions :

- le même vol fait avec un cap initial de 0° puis de 90°, avec exactement les mêmes erreurs IMU, doit donner les mêmes erreurs de navigation tournées de 90°. Cela vérifie d'un coup les conventions de repère du générateur et du strapdown ;
- doubler les biais doit doubler l'erreur qu'ils causent (régime linéaire).

## 5. Tests de mutation

`scripts/mutation_check.py` introduit 25 bugs plausibles, un par un, dans une copie du dépôt : un signe dans la matrice de transition, une correction d'attitude injectée dans le mauvais repère, une densité de bruit appliquée en `dt` au lieu de `√dt`, la gravité de signe inversé, un échantillon IMU de décalage... Il lance ensuite la suite de tests sur chaque copie. Une suite de tests qu'on n'a jamais vue échouer sur un bug connu ne prouve pas grand-chose ; c'est le test des tests. Le rapport est dans [`mutation_report.md`](mutation_report.md).

La première exécution a montré deux faiblesses réelles :

- **M16, densité de bruit gyro mal appliquée dans le générateur IMU en flux :** aucun test ne l'a vue. Ce générateur, utilisé par tous les Monte-Carlo des étapes 3 et 4, n'avait aucun test direct, et l'erreur de bruit gyro était noyée sous les biais dans les tests de cohérence.
- **M03, bruit de processus gyro en `dt²` :** il n'a été détecté que par hasard, par un test de l'étape 4 qui ne le visait pas, après 5 minutes de suite complète.

J'ai ajouté un test qui confronte le générateur à sa spécification, et une variante « bruit seul » de la comparaison covariance/Monte-Carlo. Les deux mutations sont maintenant détectées directement.

Deux mutations survivent, et c'est attendu :

- **M06, forme de Joseph remplacée par la forme courte :** c'est un mutant équivalent. Avec le gain optimal, les deux formes sont égales en arithmétique exacte.
- **M05, signe de la jacobienne de réinitialisation :** c'est un effet du second ordre, inférieur aux tolérances.

Le script s'arrête en erreur si une mutation survit sans être marquée comme attendue. En CI, il tourne chaque semaine et à la demande (environ 15 minutes).

## 6. Documentation et résultats

- **Provenance :** chaque fichier de résultats porte l'empreinte du code qui l'a produit. `scripts/check_provenance.py` et un test signalent tout résultat périmé, y compris le rapport de mutation, dont l'empreinte couvre les sources et les tests.
- **Chiffres du README :** les tableaux de résultats du README sont marqués avec leur fichier source (`<!-- source: docs/stepN_results.md -->`). `tests/test_docs.py` vérifie que chaque nombre de ces tableaux se retrouve dans le fichier généré, à l'arrondi affiché près. Changer 0,17 en 0,71 dans le README fait échouer la suite.
- **Images et arborescence :** les images citées et les fichiers listés dans l'arborescence doivent exister.

## 7. Analyse statique et CI

ruff tourne avec des règles orientées bugs : noms non définis ou inutilisés, `zip` sans `strict=`, variables de boucle inutilisées. Il a trouvé un cas limite réel. La fonction qui repère les virages pour les figures perdait silencieusement un virage encore en cours à la fin du vol, parce que `zip` tronquait la liste des débuts. C'est corrigé et vérifié.

La CI GitHub lance ruff, puis toute la suite avec la couverture de code, puis la vérification de provenance, à chaque push.

## Ce que ces tests ne couvrent pas

- **Le monde réel.** La vérité terrain et le filtre partagent les mêmes hypothèses : Terre plate, pas de rotation terrestre, pas de vent, erreurs capteurs gaussiennes. Les tests vérifient que le code est cohérent avec ce modèle, pas que le modèle décrit un vrai drone. Seuls des logs de vol réels peuvent le dire (étape 7).
- **Les tolérances.** Elles sont choisies à la main. Une tolérance trop large laisse passer un bug ; les tests de mutation sont là pour le mesurer, mais seulement sur les bugs qu'on a pensé à y mettre.
- **Les tests statistiques.** Ils utilisent des graines fixes, donc ils sont reproductibles. Avec une autre graine, un test cohérent peut échouer rarement, par construction (un intervalle à 95 % est dépassé une fois sur vingt).
- **La relecture.** Ces couches réduisent le risque qu'un bug passe, elles ne remplacent pas la relecture par quelqu'un qui comprend les équations.
