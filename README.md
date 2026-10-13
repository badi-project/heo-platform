# heo-platform — conception assistée d'oxydes à haute entropie

[![CI](https://github.com/badi-project/heo-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/badi-project/heo-platform/actions/workflows/ci.yml)

Chaîne complète allant de la récolte de littérature à une application de design
autonome : extraction d'une base de données d'oxydes à haute entropie (HEO) à partir
de 2 189 notices bibliographiques, calcul de 134 descripteurs compositionnels,
classifieur de famille structurale validé par groupes chimiques, modèle géométrique de
paramètre de maille, simulation de diffractogramme de poudre, et criblage de
734 208 compositions.

## Résultats principaux

| Sortie | Statut | Performance mesurée |
|---|---|---|
| Famille structurale (6 classes) | opérationnel | exactitude 83,7 %, F1 macro 0,734 |
| Pérovskite ou non | opérationnel | 94,8 % (règle de Goldschmidt optimisée : 86,8 %) |
| Paramètre de maille | opérationnel, ± 2 % | écart absolu moyen 1,9 % |
| Diffractogramme simulé | opérationnel | positions et intensités validées contre `pymatgen` |
| Domaine d'applicabilité | opérationnel | seuil au 95ᵉ centile des distances d'entraînement |
| Monophasé / multiphasé | peu fiable | 416 positifs contre 54 négatifs (biais de publication) |
| Propriétés catalytiques | non implémenté, délibérément | voir §6 du rapport |

Base curée : **1 133 échantillons**, 926 publications identifiées par DOI, 559 systèmes
cationiques distincts, 54 cations. Criblage : **219 242** compositions dans le domaine
d'applicabilité, dont **218 987** absentes de la base.

![Validation du classifieur](figures/fig4_validation.png)

## Application

`app/heo_designer.html` est un fichier unique sans dépendance réseau : il embarque la
base, le modèle exporté, le calcul des descripteurs et la simulation DRX. Ouvrir par
double-clic, ou cloner puis servir le dossier `app/` :

```bash
git clone https://github.com/badi-project/heo-platform.git
cd heo-platform/app && python -m http.server 8000   # puis http://localhost:8000/heo_designer.html
```

Le portage JavaScript des descripteurs est vérifié contre `features.py` sur les
200 compositions du jeu de contrôle : écart maximal de probabilité 1,1 × 10⁻¹⁶
(`QC/parity_check.txt`, régénéré par `tools/parity_check.py` et vérifié à chaque commit).

L'onglet « Candidats criblés » montre les **3 589 candidats exportés** — un
sous-ensemble plafonné à 700 par prototype, pour que les pérovskites n'écrasent pas la
liste. Le criblage intégral des 734 208 compositions est dans
`data/screening_all.csv.gz`, pas dans l'application.

## Organisation du dépôt

```
app/        application autonome (HTML + JS + modèle exporté)
src/        features.py — spécification des 134 descripteurs (source de vérité)
data/       base curée, descripteurs, candidats, criblage complet, corpus brut
data/reference/  tables élémentaires, prototypes DRX, constantes du modèle de maille
figures/    fig1 à fig6
docs/       rapport méthodologique et note de faisabilité
tools/      sync_app.py (ré-inline les JS dans le HTML), parity_check.py
tests/      suite pytest exécutée en intégration continue
QC/         résultat du contrôle de parité JavaScript / Python
```

`app/heo_designer.html` **inline** des copies de `heo_engine.js` et `heo_ui.js` entre
commentaires sentinelles. Les fichiers `app/*.js` sont la source ; le HTML est
régénéré par `python tools/sync_app.py`, et la CI refuse une divergence entre les deux.

| Fichier | Contenu |
|---|---|
| `data/heo_database.csv` | 1 133 échantillons curés, avec DOI et phrase source |
| `data/heo_features.csv` | matrice des 134 descripteurs |
| `data/heo_candidats.csv` | 3 589 candidats non publiés, avec voisin publié le plus proche |
| `data/screening_all.csv.gz` | les 734 208 compositions criblées |
| `data/heo_corpus.jsonl` | 2 189 notices récoltées (EuropePMC, Crossref, arXiv) |
| `data/heo_extracted.jsonl` | sortie brute d'extraction, avant curation |
| `data/heo_audit.csv` | audit manuel champ par champ (n = 10) |
| `app/heo_model.json` | export du modèle et jeu de contrôle de parité |

## Méthode, en bref

- **Corpus** : EuropePMC (561), Crossref (1 527), arXiv (101), filtrage par expression
  exacte sur titre et résumé, déduplication par DOI.
- **Extraction** : modèle de langue sous schéma JSON imposé, phrase source conservée.
- **Curation** : normalisation des formules, harmonisation du vocabulaire de phase,
  minimum de cinq cations principaux. Les fractions stœchiométriques extraites sont
  conservées mais **écartées de l'apprentissage** (exactitude 7/10 à l'audit) ; les
  descripteurs sont calculés en pondération équimolaire.
- **Modèle** : arbres de décision boostés par histogramme, 250 itérations, 15 feuilles.
  **Validation croisée en 20 plis groupés par système cationique** — aucun système
  chimique n'apparaît à la fois en apprentissage et en test. Probabilités calibrées par
  mise à l'échelle de température (T = 1,74 ; erreur de calibration 0,100 → 0,023).

Le détail complet, les contrôles physiques et les pièges de portage sont dans
[`docs/RAPPORT_plateforme_HEO.md`](docs/RAPPORT_plateforme_HEO.md).

## Développement

```bash
pip install -r requirements.txt     # pytest ; le code applicatif n'a aucune dépendance
python -m pytest tests -q           # 16 tests : descripteurs, données, rendu, parité
python src/features.py              # exemple : les descripteurs de (MgCoNiCuZn)O
python tools/sync_app.py --check    # le HTML est-il à jour par rapport à app/*.js ?
python tools/parity_check.py        # régénère QC/parity_check.txt (nécessite node ≥ 20)
```

Les conventions, et notamment la règle « on n'édite jamais le JavaScript inliné dans le
HTML », sont dans [CONTRIBUTING.md](CONTRIBUTING.md).

## Limites à lire avant usage

- La probabilité affichée est celle d'une **famille structurale**, jamais d'une
  performance catalytique.
- 69 % des compositions énumérées sont prédites pérovskite : l'énumération combinatoire
  sur un vivier riche en terres rares et alcalino-terreux produit massivement des
  compositions à gros cations. La liste de candidats se lit par famille chimique, pas
  ligne à ligne.
- La confusion dominante rocksalt ↔ spinelle reflète une information absente des
  résumés (température de calcination, stœchiométrie en oxygène), pas un défaut du
  classifieur.
- La confrontation expérimentale du paramètre de maille repose sur **un seul point
  indépendant**.
- Les probabilités figées dans le jeu de contrôle (`app/heo_model.json`, champ `probe`),
  produites par le modèle `scikit-learn` d'origine, se reproduisent à 9 × 10⁻⁸ près en
  médiane mais s'écartent de plus de 10⁻³ sur 22 des 200 compositions, dont une change
  de classe prédite. L'origine est connue — des seuils de coupure posés exactement sur
  des valeurs d'entraînement, où un écart d'un *ulp* fait basculer quelques arbres — et
  l'accord entre `features.py` et le moteur JavaScript reste à 10⁻¹⁶. C'est une marge à
  garder en tête pour les compositions dont les deux premières classes sont proches.
- `data/screening_all.csv.gz` (16 Mo) est versionné directement, sans Git LFS. Cela
  reste supportable à cette taille ; une reprise en LFS s'imposera si le criblage
  grossit.

## Licence

Code (`src/`, `app/`) : MIT, voir [LICENSE](LICENSE).
Données (`data/`) et figures : CC BY 4.0. Les notices bibliographiques sources restent
soumises aux conditions de leurs éditeurs respectifs ; seules les métadonnées et les
valeurs extraites sont redistribuées ici.
