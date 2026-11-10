# Journal des modifications

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Modifié — nouveau modèle de paramètre de maille

- **Rayons de Shannon par valence, coordinence et spin** (`data/reference/shannon_radii.json`,
  `src/radii.py`) : les 58 cations de la plateforme. L'ancienne table ne portait qu'une
  valence par élément. Une coordinence absente est interpolée ou extrapolée, et signalée.
- **132 mailles mesurées sourcées** (`data/reference/lattice_measured.csv`, 32 sources) :
  84 mesures d'oxydes à haute entropie des cinq familles et 48 oxydes simples. La base
  n'en comptait que 5, aucune pour les pérovskites, fluorines et pyrochlores.
- **Modèle de maille refait** (`src/lattice.py`, `latticeModel()` dans la page) : valences
  par électroneutralité, sites par classe chimique ou préférence de site, une relation
  publiée par prototype, aucune constante ajustée. Protocole écrit avant le calcul
  (`data/reference/lattice_protocol.md`). Sur 63 oxydes à haute entropie mesurés :
  **0,53 %** d'écart absolu moyen, contre 1,99 % — rocksalt 2,39 → 1,18 %, spinelle
  2,13 → 0,70 %, fluorine 1,37 → 0,27 %, pérovskite 2,39 → 0,48 %, pyrochlore
  2,25 → 0,23 % (`QC/lattice_validation.txt`).
  - **Critère de 1 % non atteint pour les rocksalt** : les compositions au Sr²⁺ et au K⁺
    ont la maille de la matrice ; la page le signale.
  - **Fluorine, validation non aveugle** : la correction des lacunes d'oxygène a été
    ajoutée après lecture des résidus ; le protocole initial donnait 1,40 %.
  - CHGNet reste utilisé pour les spinelles qu'il couvre : 0,35 % contre 0,49 % sur 13
    spinelles mesurés.
- L'incertitude affichée sur la maille est l'écart moyen de la famille (0,23 à 1,18 %) au
  lieu de 1,9 % partout. La densité suit : 1,6 % en moyenne sur 50 oxydes mesurés traités
  par le modèle géométrique (10,7 % auparavant, sur deux rocksalt).
- La colonne `a_predit` des candidats criblés est recalculée ; ni les candidats ni leurs
  probabilités ne changent.
- **Correction apprise de la maille : essayée, non intégrée**
  (`tools/learn_lattice_correction.py`, `QC/lattice_correction.txt`). Jugée hors
  échantillon, un groupe de sources retiré à chaque tour, avec des critères écrits
  d'avance : 0,53 → 0,49 %, soit 0,04 point de gain pour 0,1 exigé, et la fluorine se
  dégrade de 0,15 point. Le meilleur candidat seul, un biais par famille, atteint 0,45 %.
- Parité Python / JavaScript étendue au modèle de maille : 6 492 cas, écart nul, notes
  identiques au caractère près.

### Ajouté — la plateforme devient un outil de travail

- **Moteur de prédiction Python** (`src/predict.py`, bibliothèque standard) et **ligne de
  commande** (`tools/heo_predict.py`) : une formule ou un fichier CSV entier, sortie
  lisible, JSON ou CSV. La prédiction complète n'existait que dans la page. Les deux
  moteurs sont comparés sur 1 272 compositions : 0 désaccord, écart de probabilité
  2 × 10⁻¹⁶ ; les lignes de tableau des 1 086 formules sont identiques caractère pour
  caractère (`QC/parity_check.txt`).
- **Densité théorique** (rayons X) de la phase prédite, dans la page et en ligne de
  commande. Avec la maille mesurée de quatre oxydes de référence sourcés, écart maximal
  0,71 % (critère 1 %). Avec la maille prédite : 1,5 % sur les spinelles (critère 2 %,
  atteint), 10,7 % sur les deux rocksalt du modèle géométrique (**critère de 6 % non
  atteint**) — `tools/validate_density.py`, `QC/density_validation.txt`. L'incertitude
  affichée est la plus grande de la propagation et de l'erreur observée.
- **Onglet « Lot de formules »** : liste collée ou fichier CSV, tableau des prédictions,
  export CSV identique à celui de la ligne de commande.
- **Exports** depuis le concepteur : prédiction (.csv), profil de diffraction simulé
  (.xy), réflexions avec distances interréticulaires (.csv).
- **Comparaison à un diffractogramme mesuré** (`src/xrdfit.py`, page et `--measured`) :
  lecture des fichiers à deux colonnes et .xrdml, retrait du fond et de la raie Kα₂,
  indexation sur les cinq prototypes cubiques, affinement de la maille avec décalage de
  zéro, signalement des pics non indexés, export de l'indexation et d'une ligne pour
  `data/lab_results.csv`. Sur un jeu de motifs synthétiques jamais vu pendant la mise au
  point : maille à 0,017 % près (motifs propres), 0,14 % (motifs bruités), 4 raies
  étrangères sur 4 signalées, aucune fausse alerte (`tools/validate_xrd_fit.py`,
  `QC/xrd_fit_validation.txt`). **Validé sur motifs synthétiques seulement** : le dépôt
  ne contient aucun diffractogramme réel.

### Corrigé

- **La page arrondissait à six décimales les fractions lues dans une formule** avant de
  prédire. Sur 39 des 1 072 formules lisibles de la base, une valeur franchissait un
  seuil d'arbre et la probabilité affichée s'écartait de celle du modèle, jusqu'à
  17 points, sans changer la famille. La page lit maintenant par `readComposition()`,
  miroir de `Predictor.read()` ; vérifié dans un navigateur sur les 1 072 formules.
- **Résultat périmé** : après une formule illisible, la page laissait à l'écran le
  résultat de la formule précédente. Elle l'efface et donne la raison.
- Le choix du modèle de maille et l'enthalpie d'oxydation étaient calculés dans
  `heo_ui.js` ; ils sont dans `heo_engine.js`, seul endroit comparé à Python.

### Constaté, non corrigé

- **Modèle géométrique de maille** : sa table de rayons ne porte qu'une valence par
  élément. Sur les deux rocksalt à haute entropie de maille mesurée, l'erreur est de
  +2,3 % et −4,5 % (annoncé : 1,9 % en moyenne, spinelles compris). Aucune maille
  mesurée de pérovskite, fluorine ou pyrochlore à haute entropie ne figure dans la base :
  l'erreur du modèle n'y a jamais été mesurée. Une table de rayons par valence et par
  coordinence, validée sur des mailles mesurées sourcées, est l'étape suivante.

### Ajouté — étapes 1 à 3 : thermodynamique calculée, réductibilité, boucle active

- **`thermo/` : grandeurs physiques calculées par potentiel interatomique appris**
  (CHGNet 0.3.0), chacune confrontée à des mesures avec un critère écrit avant le
  calcul (`docs/RAPPORT_plateforme_HEO.md`, §11-§13) :
  - descripteur de monophasage des rocksalt de Sivak et al. (PRL 2025) recalculé sur
    leurs 8 compositions : 7 / 8 en un-contre-tous, enthalpie de (MgCoNiCuZn)O plus
    proche de la calorimétrie que la DFT de référence (63 contre 91 meV/atome, mesure 36) ;
  - validation redox sur six couples binaires (NIST-JANAF) et correction par élément,
    contrôlée hors échantillon (Fe₃O₄ → Fe₂O₃ : +0,67 → −0,02 eV/O) ;
  - enthalpie d'oxydation rocksalt → spinelle sur 80 systèmes de la base :
    AUC 0,95 entre systèmes rapportés rocksalt et spinelle ; aucun gain du classifieur dans son domaine (critère non atteint), 78,6 % → 81,8 % quand un cation est inconnu du modèle ;
  - paramètre de maille des spinelles calibré sur 17 binaires : 0,53 % sur 4 spinelles
    HEO mesurés (v1.0 : 1,9 %), +0,24 % sur l'échantillon interne ;
  - énergie de formation de lacune d'oxygène : corrélation de rang 0,93 avec la réductibilité mesurée de 9 oxydes binaires (critère 0,70) ; 14 spinelles HEO calculées, dont les substitutions de l'échantillon interne.
- **Carte « Thermodynamique calculée » dans l'application** : enthalpie d'oxydation et
  tendance rocksalt / spinelle pour toute composition des cations couverts, mise en avant
  hors du domaine d'applicabilité statistique ; maille et diffractogramme des spinelles
  tirés du calcul.
- **Mesure de la variance des surpotentiels OER publiés** (`thermo/analyze_oer_literature.py`) :
  corrélation intraclasse 0,00 [0,00-0,16] — aucun prédicteur ne peut être entraîné
  sur ces valeurs.
- **Boucle d'apprentissage actif** (`tools/active_learning.py`) : 27 % de synthèses en moins que le hasard sur un banc rétrospectif de 10 découpages (critère 25 %, atteint de justesse) ;
  propositions dans `data/next_experiments.csv`.
- **Résultats internes** : `data/lab_results.csv` (gabarit
  `data/lab_results_template.csv`) entre dans la base par le même lecteur de formules,
  y compris les lots multiphasés.

### Modifié — v1.1 : stœchiométrie relue dans les formules

- **Les descripteurs sont calculés sur la stœchiométrie écrite**, plus sur un mélange
  équimolaire de l'ensemble de cations du modèle de langue. `src/formula.py` lit les
  groupes, fractions, fractions rationnelles, variables (x = 0,1), placeholders
  (`M = Dy, Ho, Er`), sommes d'oxydes, composites, dopants et l'ABO₃ écrit sans
  parenthèses ; les groupes ambigus sont tranchés par bilan de charge. Miroir
  JavaScript (`FX` dans `app/heo_engine.js`) vérifié sur les 1 086 formules.
- **Base reconstruite par `tools/clean_database.py`** depuis
  `data/heo_database_raw.csv` (la v1.0, jamais modifiée) : 1 074 entrées conservées,
  142 ensembles de cations corrigés dont 121 sites restitués, 59 entrées écartées avec
  leur raison. L'ensemble d'origine reste dans la colonne `cations_llm`.
- **Chaîne d'entraînement versionnée** (`tools/train_model.py`) : validation groupée,
  temporelle (≤ 2024 → 2025-2027) et par chimie inédite, références (présence des
  cations, système le plus proche, motif stœchiométrique seul, 9 descripteurs
  classiques, régression logistique), calibration évaluée sur des systèmes non vus,
  export bit-exact. Exactitude groupée 0,837 → 0,916.
- **Criblage, figures et textes de la page régénérés** par `tools/screen.py`,
  `tools/make_figures.py` et `tools/build_app.py` ; plus aucun chiffre tapé à la main
  dans l'interface. La CI vérifie que base, page et contrôles sont la sortie exacte de
  la chaîne.
- **Saisie de formule dans l'application** : on écrit la formule comme dans un article.

### Supprimé

- Les résumés d'éditeurs de `data/heo_corpus.jsonl` (redistribués à tort sous
  CC BY 4.0) ; seule leur longueur reste (`abstract_chars`).

### Corrigé

- **Parité Python / JavaScript rompue sous CPython < 3.12.** `src/features.py`
  s'appuyait sur `sum()` de CPython pour la sommation compensée de Neumaier — une
  garantie qui n'existe que depuis la 3.12. Sous 3.11 (toujours le défaut sur de
  nombreux systèmes), les moyennes pondérées dérivaient de ~10⁻¹³ par rapport au
  moteur JavaScript, suffisant pour faire basculer des dizaines d'arbres sur 1500 et
  changer une probabilité prédite de plusieurs dizaines de points — exactement le
  mode de défaillance déjà décrit au §8 du rapport, mais côté Python cette fois.
  `tests/test_app.py::test_python_javascript_parity` échouait silencieusement dès
  qu'on l'exécutait sous 3.11. `features.py` implémente maintenant lui-même la
  sommation compensée (`nsum`/`wsum`, miroir exact de `heo_engine.js`), si bien que
  la parité ne dépend plus de la version de l'interpréteur. La matrice CI couvre
  désormais 3.10 à 3.13 pour verrouiller la régression.
- **Liens DOI morts dans l'interface.** 58 des 1 133 entrées de la base sont des
  préprints arXiv sans DOI ; le tableau « systèmes voisins » affichait pour elles un
  lien `https://doi.org/NaN`. Elles sont désormais rendues « sans DOI ».
- **`features.load_elements()` inutilisable tel quel.** Le chemin par défaut pointait
  vers `element_properties.json` dans le répertoire courant. Il est maintenant résolu
  relativement au fichier source, et `python src/features.py` affiche un exemple.
- **Table élémentaire incomplète.** `data/reference/element_properties.json` ne
  contenait pas `r_shannon_VIII` ni `r_shannon_XII`, pourtant utilisés par 10 des 134
  descripteurs : toute exécution de `features.py` échouait. Les deux propriétés ont été
  restituées depuis la charge de données de l'application, qui fait foi.
- **Liens externes** : ajout de `rel="noreferrer"` à côté de `noopener`.

### Ajouté

- **Intégration continue** (`.github/workflows/ci.yml`) : tests sur Python 3.12 et 3.13,
  contrôle de synchronisation du HTML, contrôle de parité Python / JavaScript.
- **Tests** (`tests/`, 16 cas) : longueur et ordre du vecteur de descripteurs, entropie
  configurationnelle, facteur de Goldschmidt de SrTiO₃, normalisation des sous-réseaux
  A/B, complétude de la table élémentaire, forme des fichiers de données, rendu des
  entrées sans DOI.
- **`tools/sync_app.py`** : les copies de `heo_engine.js` et `heo_ui.js` inlinées dans
  `heo_designer.html` sont délimitées par des sentinelles et régénérées depuis les
  fichiers source. La divergence silencieuse entre les deux copies est maintenant
  détectée par la CI.
- **`tools/parity_check.py`** : le contrôle de parité, jusqu'ici figé dans un fichier
  texte, est régénérable et exécuté à chaque commit. Il a mis au jour un écart reproductible
  entre le modèle exporté et les probabilités figées du jeu de contrôle (22 compositions
  sur 200 au-delà de 10⁻³, un changement de classe) ; il est documenté dans les limites du
  README et n'est pas bloquant, l'accord Python / JavaScript restant à 10⁻¹⁶.
- `CONTRIBUTING.md`, modèles d'issue et de *pull request*, ce journal.

### Modifié

- Le compteur du tableau de candidats indique que seules les 400 premières lignes du
  tri courant sont affichées, et que les 3 589 candidats exportés sont un
  sous-ensemble du criblage de 734 208 compositions.
- Le tableau « systèmes voisins » rappelle que la colonne « phase » reflète un corpus
  de 416 lots monophasés contre 54 multiphasés.

## [1.0.0] — 2026-10-13

Première publication : base de 1 133 échantillons extraite de 2 189 notices,
134 descripteurs, classifieur de famille structurale (exactitude 83,7 %), modèle
géométrique de paramètre de maille, simulation de diffractogramme, criblage de
734 208 compositions et application autonome `heo_designer.html`.
