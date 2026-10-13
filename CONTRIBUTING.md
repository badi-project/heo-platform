# Contribuer

## Mise en route

```bash
git clone https://github.com/badi-project/heo-platform.git
cd heo-platform
python -m pip install -r requirements.txt      # pytest uniquement
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

## Ajouter des données

Ajouter des lignes à `data/heo_database.csv` en respectant les colonnes `cations`,
`n_cat`, `phase` (`single`/`multi`), `proto`, `a_ang`, `T_C`, `route`, `doi` — une
référence interne de cahier de laboratoire convient si le résultat n'est pas publié.
Les lots multiphasés comptent autant que les réussites : la base manque d'exemples
négatifs. Réentraîner **toujours** avec une validation groupée par système cationique,
jamais avec un découpage aléatoire.

## Style

Python : PEP 8, lignes de 100 colonnes au plus, bibliothèque standard uniquement dans
`src/`. JavaScript : pas de dépendance externe, pas d'étape de construction autre que
`tools/sync_app.py`. Les messages de commit décrivent l'effet observable, pas le fichier
touché.
