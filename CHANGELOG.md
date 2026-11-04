# Journal des modifications

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

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
