# Contribuer

## Mise en route

```bash
git clone https://github.com/badi-project/heo-platform.git
cd heo-platform
python -m pip install -r requirements.txt      # pytest uniquement
# pour réentraîner : python -m pip install -r requirements-train.txt
python -m pytest tests -q
```

Node 20 ou plus récent est nécessaire pour les deux contrôles qui exécutent le moteur
JavaScript (`tools/parity_check.py`, un test de `tests/test_app.py`). Sans node, ces
tests sont ignorés automatiquement ; l'intégration continue, elle, les exécute.

## Règle à ne pas contourner : `app/*.js` est la source

`app/heo_designer.html` est volontairement autonome — il inline la charge de données, le
moteur et l'interface. Les copies inlinées de `heo_engine.js` et `heo_ui.js` sont
délimitées par des commentaires sentinelles et **régénérées**, jamais éditées à la main :

```bash
# éditer app/heo_engine.js ou app/heo_ui.js, puis
python tools/sync_app.py          # réinjecte dans heo_designer.html
python tools/sync_app.py --check  # ce que vérifie la CI
```

Une modification du HTML qui ne passe pas par les fichiers source fait échouer la CI.

## Modifier les descripteurs

`src/features.py` est la seule source de vérité du vecteur de 134 descripteurs ;
`app/heo_engine.js` en est un miroir vérifié numériquement. Toute modification doit être
portée des deux côtés, puis :

```bash
python tools/parity_check.py      # régénère QC/parity_check.txt
```

Le portage doit être **bit-exact**, pas seulement proche : les seuils de coupure du
modèle sont posés sur des valeurs d'entraînement, et un écart d'un *ulp* suffit à faire
basculer des dizaines d'arbres. Les pièges déjà rencontrés (arrondi des seuils à
l'export, `Math.fround` superflu, sommation compensée de Neumaier de CPython) sont
documentés au §8 de `docs/RAPPORT_plateforme_HEO.md`.

Changer les descripteurs invalide le modèle entraîné : il faut réexporter
`app/heo_model.json`, avec des seuils non arrondis.

## Ajouter des données et régénérer

`data/heo_database.csv` est une **sortie** de `tools/clean_database.py` : ne pas
l'éditer. Les résultats de laboratoire s'ajoutent dans `data/lab_results.csv` (gabarit :
`data/lab_results_template.csv`), une ligne par synthèse : identifiant de cahier, date,
`formula_text` (la formule **telle qu'écrite**, sites compris : `La(CrMnFeCoNi)O3`, pas
`La-Cr-Mn-Fe-Co-Ni`), `proto`, `phase` (`single`/`multi`), température, atmosphère,
maille mesurée. Les lots multiphasés comptent autant que les réussites : la base manque
d'exemples négatifs. `data/heo_database_raw.csv` est la v1.0 publiée et ne se modifie
pas.

La chaîne se rejoue dans cet ordre (`pip install -r requirements-train.txt`) :

```bash
python tools/clean_database.py    # formules -> fractions cationiques, QC/database_cleaning.txt
python tools/train_model.py       # descripteurs, 3 validations, export, QC/training_report.txt
python tools/screen.py            # criblage et candidats, QC/screening_report.txt
python tools/build_app.py         # charge de données et texte « Méthode » de la page
python tools/make_figures.py      # figures 4 et 5
python tools/parity_check.py      # parité Python / JavaScript, QC/parity_check.txt
```

Réentraîner **toujours** avec une validation groupée par système cationique, jamais
avec un découpage aléatoire. `train_model.py` refuse d'écrire un export qui ne
reproduit pas scikit-learn à l'identique, et `screen.py` refuse de cribler avec un
modèle qui ne reproduit pas l'export publié.

## Calculs thermodynamiques (`thermo/`)

Les calculs CHGNet prennent des heures ; leurs résultats sont versionnés dans
`data/thermo/` et la CI ne les relance pas : elle vérifie que les rapports, les
substituts et la page en sont la sortie (`tests/test_thermo.py`). Après un nouveau
calcul, relancer l'analyse correspondante puis `tools/build_app.py`. Toute nouvelle
grandeur calculée arrive avec son critère d'acceptation écrit avant le calcul et sa
confrontation à des mesures sourcées (`data/thermo/reference_thermo.csv`), comme les
autres (`thermo/README.md`).

## Modifier l'analyseur de formules

`src/formula.py` est la référence ; `FX` dans `app/heo_engine.js` en est le miroir.
Toute nouvelle notation s'ajoute d'abord comme cas dans `tests/test_formula.py`, puis
des deux côtés. `tools/parity_check.py` compare les deux sur toutes les formules de la
base.

## Style

Python : PEP 8, lignes de 100 colonnes au plus, bibliothèque standard uniquement dans
`src/`. JavaScript : pas de dépendance externe, pas d'étape de construction autre que
`tools/sync_app.py`. Les messages de commit décrivent l'effet observable, pas le fichier
touché.
