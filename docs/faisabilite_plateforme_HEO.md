# Plateforme de conception in silico d'oxydes à haute entropie (HEO) — note de faisabilité

## 1. Réponse courte

Oui pour une partie du problème, non pour l'autre, et la frontière entre les deux est
nette :

| Ce que tu veux prédire | Faisable à partir de la littérature ? | Pourquoi |
|---|---|---|
| **Phase / structure** (monophasé ou non, rocksalt / spinelle / pérovskite / fluorine / bixbyite) | **Oui** — c'est déjà fait, précision 85–98 % | L'étiquette est catégorielle, lisible sur un DRX, et rapportée systématiquement dans chaque article |
| **Paramètre de maille, diffractogramme simulé, distribution des longueurs de liaison** | **Oui** | Calculable (pas besoin d'apprendre) : structure modèle + potentiel interatomique ou DFT |
| **Enthalpie de mélange, tendance à la ségrégation, descripteur d'entropie (EFA)** | **Oui** | Calculable ab initio, à coût raisonnable avec les potentiels ML universels |
| **Morphologie / taille de particule / surface spécifique** | **Non** | Déterminées par la voie de synthèse (SPS, Pechini, combustion, pyrolyse d'aérosol…), pas par la composition |
| **Activité électrocatalytique (η à 10 mA/cm², pente de Tafel)** | **Classement qualitatif au mieux** | L'étiquette publiée n'est pas une propriété du matériau mais celle d'un *électrode* (charge, substrat, liant, électrolyte, normalisation, correction iR) |
| **Activité photocatalytique (k apparent, % dégradation)** | **Non, pas quantitativement** | Encore plus dépendant du protocole : lampe, filtre, concentration du polluant, masse de catalyseur, volume, scavengers. `Eg` est en revanche prédictible |

Autrement dit : **un logiciel de pré-criblage structural et thermodynamique est
réaliste et utile ; un simulateur qui sort un surpotentiel OER fiable pour une
composition inédite n'est pas atteignable avec la littérature seule.** La bonne
façon de formuler le projet est donc « filtre amont » et non « remplacement de la
manip ».

## 2. Ce qui existe déjà (à ne pas refaire)

Le concept est validé et publié, ce qui est une bonne nouvelle : cela veut dire que
le pipeline marche, et que la contribution se situe ailleurs (couverture, propriétés,
interface, boucle expérimentale).

- **Extraction LLM + ML explicable sur les HEO** — *Rational Design of Single-Phase
  High-Entropy Oxides via Large Language Model Data Mining and Explainable Machine
  Learning*, J. Chem. Inf. Model. (2026), DOI 10.1021/acs.jcim.6c00752. Des agents
  LLM extraient composition / phase / méthode de synthèse depuis les résumés avec
  ~96 % d'exactitude de mise en base ; XGBoost atteint F1 ≈ 86 % sur une
  classification à 7 classes de structures, et un réseau de neurones 97,9 % pour
  « pérovskite vs non-pérovskite » — là où le facteur de tolérance de Goldschmidt
  plafonne à 67,3 %. C'est **exactement** l'architecture que tu décris, déjà publiée :
  lis-le en premier.
- **Descripteurs d'entropie ab initio** — Sarker *et al.*, Nat. Commun. **9**, 4980
  (2018) : l'*Entropy Forming Ability* (EFA), construit sur la dispersion des
  énergies de configurations désordonnées, prédit quelles compositions se stabilisent
  en solution solide monophasée. Transposable aux oxydes.
- **Potentiels interatomiques ML pour les HEO** — Phys. Rev. Lett. **134**, 216101
  (2025) : cartographie rapide de l'espace de composition des HEO rocksalt
  monophasés avec deux descripteurs (distribution des longueurs de liaison et
  enthalpie de mélange), calculés par potentiel ML au lieu de DFT. C'est le moteur
  « calcul » de ta plateforme.
- **GNN pour une propriété fonctionnelle** — *OxiGraphX*, Small (2025), DOI
  10.1002/smll.202501946 : réseau de neurones de graphe prédisant l'énergie de
  formation des lacunes d'oxygène (OVFE) dans les pérovskites à haute entropie,
  avec validation DFT puis expérimentale OER. Noter la stratégie : on n'apprend pas
  le surpotentiel directement, on apprend un **descripteur physique calculable**
  (OVFE) qui corrèle avec l'activité.
- **Référence fondatrice** : Rost *et al.*, « Entropy-stabilized oxides », Nat.
  Commun. **6**, 8485 (2015) — le (MgCoNiCuZn)O rocksalt, point de départ de tout
  le domaine.

## 3. Le vrai verrou : les données, pas les algorithmes

### 3.1 L'ordre de grandeur du problème combinatoire

Avec ~30 cations raisonnablement documentés en HEO :

```
quinaires équimolaires               C(30,5)  =        142 506
quinaires, pas de contrainte équimolaire,
résolution 5 % at.                   × C(19,4) ≈  5,5 × 10^8
+ sénaires, septénaires, anions mixtes, O non stœchiométrique → >10^10
```

### 3.2 L'ordre de grandeur des données disponibles

C'est là que ça coince. Les jeux de données HEO publiés pour l'apprentissage sont
petits : 66 échantillons extraits manuellement pour les spinelles à haute entropie
(88 % de précision de classification de phase), 34 variantes compositionnelles pour
une série de pérovskites SOFC, 381 compositions d'oxydes synthétisées de l'unaire au
quinaire dans l'étude expérimentale la plus large. Pour l'OER, un travail sur un
espace compositionnel restreint n'obtient que **R² = 0,58–0,64** ; une autre étude
agrège 63 publications pour n'obtenir que **88 instances**.

**Conclusion : ~10³ entrées atteignables pour la phase, ~10¹–10² pour n'importe
quelle propriété catalytique comparable.** Contre 10⁸ candidats. Un modèle entraîné
sur 88 points d'OER n'extrapole pas vers une chimie inédite — il interpole dans la
famille dont il vient.

### 3.3 Les quatre biais qui vont te mordre

1. **Biais de publication / absence de négatifs.** Personne ne publie
   « (FeCoNiCrMn)₃O₄ calciné à 900 °C donne trois phases ». Ton jeu de données est
   donc massivement déséquilibré vers le monophasé. C'est un problème connu et
   traité (sur-échantillonnage Borderline-SMOTE / ADASYN pour les carbures à haute
   entropie, npj Comput. Mater. 2026, DOI 10.1038/s41524-025-01873-2), mais une
   classe négative synthétique reste une béquille.
2. **La composition ne détermine pas la phase à elle seule.** La même composition
   est monophasée ou multiphasée selon T de calcination, rampe, trempe, atmosphère,
   taille de grain. Un modèle qui ne prend que la composition en entrée apprend en
   réalité « cette composition *telle qu'elle est habituellement synthétisée* ». Il
   **faut** mettre la température et la voie de synthèse dans les descripteurs.
3. **Incomparabilité des mesures fonctionnelles.** Deux η à 10 mA/cm² issus de deux
   labos diffèrent de 100 mV pour des raisons purement protocolaires. Tu as un
   dossier `substrat` dans tes propres données : tu sais déjà à quel point le
   substrat déplace la réponse. Mélanger ces valeurs dans un même régresseur, c'est
   apprendre du bruit de protocole.
4. **Fuite d'information par famille.** Si tu splittes aléatoirement, les 12
   variantes de (MgCoNiCuZn)O se retrouvent des deux côtés et ton score explose
   artificiellement. **Il faut une validation croisée par système chimique
   (leave-one-system-out)**, pas aléatoire. C'est le point sur lequel un rapporteur
   t'attendra.

## 4. Architecture proposée

Cinq couches, chacune livrable et testable indépendamment.

### Couche 1 — Corpus et extraction

- Récupération des métadonnées et résumés : API Crossref, OpenAlex, PubMed/EuropePMC,
  arXiv ; PDF/XML pour l'accès complet là où c'est permis.
- Extraction structurée par LLM avec **schéma de sortie imposé** (JSON strict), et
  non en texte libre : `{formule brute, cations + stœchiométries, anions, phase(s)
  DRX, groupe d'espace, paramètre de maille, voie de synthèse, T/t de calcination,
  atmosphère, surface BET, Eg, DOI}`.
- Normalisation chimique : parsing des formules avec `pymatgen.Composition`,
  résolution des notations `(A₀.₂B₀.₂…)O`, inférence des proportions relatives quand
  l'article écrit seulement « equimolar ».
- **Règle non négociable : chaque champ porte son DOI et sa phrase source.** Sans
  traçabilité, la base est inutilisable pour une publication et impossible à
  déboguer.
- Validation : annotation manuelle de 100 articles tirés au hasard → précision /
  rappel par champ. C'est la mesure qui crédibilise tout le reste.

### Couche 2 — Descripteurs

Pour chaque composition, calcul automatique (`matminer`, `pymatgen`) de :

- **Géométriques** : rayons ioniques de Shannon pondérés, écart-type des rayons (δr),
  facteur de tolérance de Goldschmidt et facteur octaédrique pour les pérovskites,
  somme des rayons cationiques.
- **Électroniques** : électronégativité (Pauling et Sanderson) moyenne et variance,
  VEC, états d'oxydation moyens et mixité rédox, enthalpie d'atomisation.
- **Thermodynamiques** : ΔS_config, ΔH_mix estimée, Ω = TΔS/|ΔH|.
- **Statistiques** : moyenne / écart-type / min / max / étendue de chaque propriété
  élémentaire — c'est la variante de base de données qui s'est révélée optimale dans
  le papier JCIM.
- **Procédé** : T de calcination, durée, atmosphère, voie de synthèse encodée.

L'analyse SHAP sur ce type de jeu montre que la stabilité des pérovskites à haute
entropie est gouvernée par le couplage entre facteurs géométriques (somme des rayons
cationiques) et descripteurs électroniques (électronégativité de Sanderson, enthalpie
d'atomisation) — donc garde les deux familles.

### Couche 3 — Prédiction de phase (le cœur fiable)

- Classification multiclasse (rocksalt / spinelle / pérovskite / fluorine / bixbyite /
  multiphasé) : XGBoost ou LightGBM, + un petit MLP en comparaison.
- Gestion du déséquilibre : poids de classes, Borderline-SMOTE, et **report de la
  matrice de confusion, pas seulement de l'exactitude globale**.
- Calibration des probabilités (isotone ou Platt) : l'utilisateur a besoin de
  « 72 % monophasé » et non d'un verdict binaire.
- Comparaison obligatoire à la ligne de base physique (Goldschmidt, δr, règles de
  Hume-Rothery transposées). Le gain par rapport au descripteur classique *est* le
  résultat.
- **Domaine d'applicabilité** : distance au plus proche voisin du jeu
  d'entraînement dans l'espace des descripteurs, affichée avec chaque prédiction.
  Hors domaine → refuser de prédire plutôt que mentir.

### Couche 4 — Structure et signature observable (la partie calculée, pas apprise)

C'est la couche qui donne le plus de valeur par unité d'effort, parce qu'elle ne
dépend pas de la qualité de la base :

1. Génération d'une supercellule désordonnée représentative : SQS (`icet`, ATAT
   `mcsqs`) pour la structure cible prédite, ou tirages Monte-Carlo d'occupation.
2. Relaxation avec un potentiel interatomique ML universel (**MACE-MP**, **CHGNet**,
   **M3GNet**) — secondes à minutes sur un portable, là où la DFT demanderait des
   heures de cluster.
3. Sorties directement comparables à la manip : **paramètre de maille**,
   **diffractogramme simulé** (`pymatgen.analysis.diffraction.xrd.XRDCalculator`),
   distribution des longueurs de liaison, enthalpie de mélange, énergie de formation
   de lacune d'oxygène, DOS / gap (avec DFT ponctuelle pour ce dernier).
4. Vérification DFT (VASP / Quantum ESPRESSO) sur les 5–10 candidats finaux
   seulement.

Un étudiant qui sort un diffractogramme prédit, superposable à son DRX mesuré, a
quelque chose de vérifiable — c'est beaucoup plus solide qu'un chiffre d'activité
non falsifiable.

### Couche 5 — Conception inverse et boucle active

- Criblage direct : énumération de l'espace quinaire/sénaire sur les cations choisis,
  filtrage par probabilité de monophase × domaine d'applicabilité × contraintes
  utilisateur (pas de Co, pas de terre rare, coût, toxicité).
- **Nouveauté** : vérification automatique que le candidat n'est pas déjà dans la
  base (et donc déjà publié) — un vrai service rendu.
- Optimisation bayésienne / apprentissage actif : le modèle ne propose pas « le
  meilleur », il propose **les 5 synthèses les plus informatives**. Les résultats de
  manip (y compris les échecs !) retournent dans la base. C'est le seul mécanisme qui
  fait progresser un modèle entraîné sur 10² points.
- Les propriétés fonctionnelles sortent en **rang relatif avec incertitude**
  (« ce candidat est probablement dans le quartile supérieur pour l'OER parmi les
  spinelles Fe-Co-Ni »), jamais en valeur absolue de η.

### Pile technique

| Besoin | Outil |
|---|---|
| Prototype en 2 semaines | Streamlit ou Gradio |
| Version pérenne | FastAPI + React, base PostgreSQL |
| Chimie / structures | pymatgen, ASE, matminer, icet |
| Potentiels ML | MACE-MP, CHGNet, M3GNet |
| ML tabulaire | scikit-learn, XGBoost, SHAP |
| Visualisation 3D de la maille | 3Dmol.js ou NGL |
| Base + code ouverts | dépôt Zenodo versionné + GitHub |

## 5. Périmètre réaliste pour un stage M2 (≈ 6 mois)

Ne construis pas la plateforme complète. Construis **un bout vérifiable**, avec la
base de données comme livrable principal — c'est elle qui sera citée.

- **Mois 1** : lecture (commence par le papier JCIM), schéma de base de données,
  200 articles annotés à la main → jeu de référence pour évaluer l'extraction.
- **Mois 2** : pipeline d'extraction sur ~1 500–3 000 résumés, rapport de
  précision/rappel par champ, base nettoyée et versionnée.
- **Mois 3** : descripteurs + classifieur de phase, validation
  leave-one-system-out, comparaison à Goldschmidt, analyse SHAP.
- **Mois 4** : couche calcul — SQS + MACE/CHGNet → paramètre de maille et DRX
  simulé ; **validation sur tes propres échantillons déjà caractérisés** (dossiers
  `avant stage`, `apres stage electro`, `apres stage photo`). Ça, c'est ta vraie
  carte : la plupart des papiers de ce type n'ont aucune validation expérimentale
  propre.
- **Mois 5** : interface Streamlit — l'utilisateur coche 5 à 8 cations, obtient
  probabilité de phase, maille, DRX simulé, voisins publiés les plus proches, et un
  drapeau « hors domaine » le cas échéant.
- **Mois 6** : proposition de 3 candidats, synthèse, DRX, comparaison
  prédiction/mesure. Même un échec est publiable s'il est analysé.

Restreindre à **une famille structurale** (les spinelles, par exemple, si c'est ce
que tu synthétises) multiplie par trois les chances que le modèle soit bon. Le
généraliste sur toutes les structures sera médiocre partout.

## 6. Limites à annoncer d'emblée

- La plateforme ne prédira pas la **voie de synthèse** qui mène au monophasé. C'est
  une limitation structurelle, pas un manque d'effort.
- Elle ne traitera ni la **stabilité en opération** (dissolution, ségrégation de
  surface, reconstruction en (oxy)hydroxyde sous OER — qui est souvent *la* phase
  active réelle), ni le vieillissement.
- La surface d'un HEO en fonctionnement n'a pas la composition du bulk. Un modèle
  entraîné sur la composition nominale prédit donc une propriété d'un objet qui
  n'existe plus après 30 minutes de chronopotentiométrie.
- Les valeurs photocatalytiques de la littérature sont, en pratique, non agrégeables.
  Soit tu te limites à `Eg` et à la position des bandes (prédictibles), soit tu
  construis ton propre jeu de données à protocole fixe.
- Le périmètre honnête, donc : **réduire 10⁸ candidats à 20 qui valent une
  semaine de four.** C'est déjà énorme.

## 7. Références de départ

1. Rational Design of Single-Phase High-Entropy Oxides via LLM Data Mining and
   Explainable Machine Learning — *J. Chem. Inf. Model.* (2026).
   DOI 10.1021/acs.jcim.6c00752 — **le plus proche de ton idée, à lire en premier.**
2. Discovering High-Entropy Oxides with a Machine-Learning Interatomic Potential —
   *Phys. Rev. Lett.* **134**, 216101 (2025).
3. Sarker *et al.*, High-entropy high-hardness metal carbides discovered by entropy
   descriptors — *Nat. Commun.* **9**, 4980 (2018).
4. Kaufmann *et al.*, Discovery of high-entropy ceramics via machine learning —
   *npj Comput. Mater.* **6**, 42 (2020).
5. Machine Learning-Guided Discovery of High-Entropy Perovskite Oxide
   Electrocatalysts via Oxygen Vacancy Engineering (OxiGraphX) — *Small* (2025).
   DOI 10.1002/smll.202501946.
6. Machine learning for phase prediction of high-entropy carbide ceramics from
   imbalanced data — *npj Comput. Mater.* (2026). DOI 10.1038/s41524-025-01873-2.
7. Chen *et al.*, A map of single-phase high-entropy alloys — *Nat. Commun.* **14**,
   2856 (2023).
8. Rost *et al.*, Entropy-stabilized oxides — *Nat. Commun.* **6**, 8485 (2015).
