# heo-platform — conception assistée d'oxydes à haute entropie

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

Le portage JavaScript des descripteurs est vérifié bit-à-bit contre `features.py` sur
200 compositions tirées au hasard : écart maximal de probabilité 5,6 × 10⁻¹⁶
(`QC/parity_check.txt`).

## Organisation du dépôt

```
app/        application autonome (HTML + JS + modèle exporté)
src/        features.py — spécification des 134 descripteurs (source de vérité)
data/       base curée, descripteurs, candidats, criblage complet, corpus brut
data/reference/  tables élémentaires, prototypes DRX, constantes du modèle de maille
figures/    fig1 à fig6
docs/       rapport méthodologique et note de faisabilité
QC/         contrôle de parité JavaScript / Python
```

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

## Licence

Code (`src/`, `app/`) : MIT, voir [LICENSE](LICENSE).
Données (`data/`) et figures : CC BY 4.0. Les notices bibliographiques sources restent
soumises aux conditions de leurs éditeurs respectifs ; seules les métadonnées et les
valeurs extraites sont redistribuées ici.
