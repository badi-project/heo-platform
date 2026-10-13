# Journal des modifications

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Corrigé

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
