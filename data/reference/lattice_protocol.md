# Protocole de validation du modèle de maille

Ce protocole est écrit avant tout calcul du nouveau modèle. Les données, la séparation
calibration / validation, les règles du modèle et les critères de réussite y sont
fixés. Une règle modifiée après la lecture des résidus des oxydes à haute entropie est
consignée dans le rapport (`QC/lattice_validation.txt`), et la famille concernée est
alors déclarée « validation non aveugle ».

## 1. Données

`data/reference/lattice_measured.csv` contient une maille mesurée par ligne. Chaque ligne
porte sa source (DOI, identifiant arXiv ou URL) et l'endroit précis où se trouve la
valeur (tableau, texte, information complémentaire).

| colonne | contenu |
|---|---|
| `id` | identifiant stable (RS, SP, FL, PV, PY : oxydes à haute entropie ; RF : références) |
| `group` | `heo` pour la validation, `reference` pour les oxydes simples et les solutions à 2 ou 3 cations |
| `proto` | rocksalt, spinel, fluorite, perovskite, pyrochlore |
| `formula_text` | formule telle qu'écrite dans la source, lue par `src/formula.py` |
| `a_ang` | paramètre cubique (Å) |
| `abc_ang` | a;b;c (Å) d'une maille orthorhombique Pbnm. Le paramètre pseudo-cubique vaut (abc/4)^(1/3) |
| `phase_note` | `single` si la source déclare une seule phase, sinon la fraction ou l'impureté |
| `synthesis`, `source`, `source_detail` | traçabilité |

Toutes les mailles sont mesurées à température ambiante.

## 2. Séparation

- **Références** (`group = reference`) : oxydes simples et solutions à 2 ou 3 cations.
  Elles servent à vérifier les relations publiées et la table de rayons, et à fixer
  l'ordre de préférence des sites du spinelle (section 4). Aucun paramètre numérique
  n'est ajusté sur elles.
- **Validation** (`group = heo`) : oxydes à haute entropie des cinq familles. Aucune
  valeur de ce groupe n'entre dans le modèle géométrique.

## 3. Unité d'évaluation

L'unité d'évaluation est la composition unique : prototype, plus fractions cationiques
arrondies à 0,001. Sa maille mesurée est la médiane de ses lignes, si bien qu'une
composition mesurée sept fois ne pèse pas sept fois.

La métrique principale ne retient que les lignes `phase_note = single`. Les autres sont
rapportées à part.

Le plancher est la dispersion entre laboratoires. Pour chaque composition mesurée par
au moins deux sources, on calcule l'écart absolu moyen à la médiane, en %. Le
plancher, rapporté pour chaque famille, est la moyenne de cet écart. Aucun modèle ne
peut descendre durablement sous ce plancher.

## 4. Règles du modèle géométrique (étape 3)

**Rayons.** Les rayons sont ceux de Shannon (1976), donnés par `src/radii.py` :
valence, coordinence, haut spin sauf Co3+ et Ni3+ (bas spin). Une coordinence absente
de la table est extrapolée par le rapport des ions de même charge ; le rapport le
signale.

**Valences.** Chaque élément a un état par défaut. On ajuste ensuite la charge totale
à la cible du prototype en oxydant les cations rédox dans un ordre fixé. Le dernier
cation de la liste peut avoir une valence fractionnaire.

- Rocksalt : cible +2 par cation.
  - Valences par défaut : 2+ ; Li, Na et K 1+ ; Ga, Al et Cr 3+.
  - Si la charge est trop basse, on oxyde dans l'ordre Co → Mn → Fe → Ni (2+ → 3+).
  - Si elle est trop haute, l'écart est porté par des lacunes et rien n'est réduit.
- Spinelle AB2O4 : cible +8 pour 3 cations.
  - Valences par défaut : Mg, Zn, Ni, Cu, Co, Fe et Mn 2+ ; Al, Cr et Ga 3+ ; Ti, Zr, Sn et Ge 4+ ; Li 1+.
  - On oxyde dans l'ordre Fe → Mn → Co (2+ → 3+).
- Fluorine : pas d'ajustement de charge ; l'écart à +4 est porté par des lacunes d'oxygène.
  - Ce, Zr, Hf, Ti, Sn 4+ ; Ta et Nb 5+ ; terres rares, Y, Fe et Bi 3+ ; Ca, Sr et Mg 2+.
  - Pr et Tb prennent la valence de leur oxyde stable à l'air : Pr6O11 et Tb4O7, soit
    Pr 3,67 (2/3 Pr4+) et Tb 3,5.
  - Mn prend celle de Mn3O4, stable aux températures de frittage (2/3 Mn3+, 1/3 Mn2+).
- Pérovskite ABO3 : cible +6.
  - Site A : 3+ pour les terres rares, Y et Bi ; 2+ pour Ba, Sr et Ca ; 1+ pour Na et K.
  - Site B : par défaut, Ti, Zr, Hf et Sn 4+ ; Ga, In, Al et Sc 3+ ; Nb et Ta 5+.
    Les métaux de transition 3d (Co, Cr, Fe, Mn, Ni, Cu) complètent la charge : 3+ si le
    site A est 3+, 4+ si le site A est 2+ et que les autres cations B ne suffisent pas.
  - Charge en défaut avec des cations B à valence fixe : lacunes d'oxygène, pas d'ajustement.
- Pyrochlore A2B2O7 : terres rares et Y 3+ sur le site A ; Ti, Zr, Hf et Sn 4+ sur le site B.

**Sites.**

- Pérovskite. Sont placés en A : terres rares, Y, Bi, Ba, Sr, Ca, Pb, Na et K. Tous les
  autres cations vont en B.
- Pyrochlore. Les terres rares et Y vont en A ; Ti, Zr, Hf et Sn vont en B.
- Spinelle. Le site tétraédrique, un cation sur trois, est rempli par score de
  préférence décroissant. Le dernier cation retenu peut n'y être que partiellement.
  Les autres cations vont en site octaédrique.

Scores de préférence tétraédrique du spinelle :

| score | cations |
|---|---|
| 10 | Zn2+ |
| 9 | Mn2+ |
| 8 | Fe3+ |
| 7,5 | Ga3+ |
| 7 | Co2+ |
| 6 | Mg2+ |
| 5 | Fe2+ |
| 4 | Cu2+ |
| 3 | Al3+ |
| 2 | Li+ |
| 1 | Mn3+, Ni2+ |
| 0 | Cr3+, Co3+, Ti4+, Zr4+, Sn4+, Ge4+ |

Cet ordre suit l'énergie de préférence octaédrique du champ cristallin, nulle pour les
ions d0, d5 et d10, et l'inversion connue des spinelles binaires de référence. Il
redonne :

- normaux : MgAl2O4, ZnAl2O4, CoAl2O4, FeAl2O4, les chromites, ZnFe2O4, MnFe2O4, Co3O4 ;
- inverses : NiAl2O4, MgFe2O4, NiFe2O4, CoFe2O4, Fe3O4, MgGa2O4, Mg2TiO4, Zn2TiO4.

Il est fixé ici, avant le calcul sur les oxydes à haute entropie.

**Relations publiées.** Aucun paramètre n'est ajusté par nous. r_O désigne le rayon de
l'oxygène, dans la coordinence indiquée.

- Rocksalt : a = 2 (r̄_M^VI + r_O^VI), avec r_O^VI = 1,40 Å (contact anion-cation).
- Fluorine : a = (4/√3)(r̄_M^VIII + r_O^IV), avec r_O^IV = 1,38 Å.
  - Limite : les lacunes d'oxygène des fluorines riches en cations trivalents sont ignorées.
- Pérovskite (Ubic 2007, doi 10.1111/j.1551-2916.2007.01881.x) :
  a_pc = 0,06741 + 0,49052 (r̄_A + r_O) + 1,29212 (r̄_B + r_O).
  - Tous les rayons sont en coordinence VI, avec r_O = 1,40 Å.
- Spinelle (O'Neill et Navrotsky 1983, Am. Mineral. 68, 181, équations 1 et 2) :
  a = (8 / (11√3)) [5 R_T + √(33 R_O² − 8 R_T²)].
  - R_T = r̄_T^IV + 1,38 Å et R_O = r̄_O^VI + 1,38 Å.
  - u est éliminé entre les deux distances.
- Pyrochlore (Mouta, Silva et Paschoal, arXiv 1212.0908) :
  a = (8/√3) [1,43373 (R_A) − 0,42931 (R_A)² / R_B].
  - R_A = r̄_A^VIII + 1,38 Å et R_B = r̄_B^VI + 1,38 Å.

r̄ est la moyenne des rayons pondérée par les fractions du site.

## 4 bis. Amendements et précisions

Tout ce qui suit a été écrit après la première version de ce protocole. Chaque entrée
dit si les résidus des oxydes à haute entropie avaient déjà été lus.

- **A1, fluorine, après lecture des résidus.** La relation de contact seule
  surestimait toutes les fluorines lacunaires, références comprises (CeO2 +0,25 %,
  La0,5Ce0,5O1,75 +1,32 %, Ce-La-Pr +1,59 %). Les lacunes d'oxygène sont désormais
  traitées en sphères dures : pour MO(2−d), le rayon anionique vaut
  ((2 − d) r_O^IV + d r_V) / 2. Le rayon r_V de la lacune dépend de l'hôte
  tétravalent : r_V = c0 − c1 (r_s − r_h), avec les constantes publiées pour HfO2,
  ZrO2 et CeO2 (Chatzichristodoulou et al., J. Electroceram. 34 (2015) 100,
  doi 10.1007/s10832-014-9916-2, tableau 1). r_s est le rayon VIII moyen des cations
  de valence inférieure à 4, pondéré par les lacunes qu'ils créent ; un hôte sans
  constante publiée prend celles de l'hôte de rayon le plus proche. Aucune constante
  n'est ajustée. **La validation de la fluorine n'est plus aveugle** ; le rapport donne
  à côté le résultat du protocole initial.
- **A2, rocksalt, sans lien avec les résidus.** Un élément sans ion 2+ dans la table de
  Shannon (Y, Sc, Zr, terres rares…) prend la valence par défaut de la plateforme au
  lieu de rester sans maille. Aucune composition de validation n'est concernée ; la
  règle sert au criblage.
- **Précision, pérovskite, écrite dans le code avant le calcul sur la validation.**
  L'ordre d'oxydation des cations B 3d est Mn → Fe → Co → Ni → Cr → Cu ; l'ordre de
  réduction, si la charge dépasse +6, est Ni → Cu → Co → Fe → Mn → Cr.
- **Précision, pérovskite et pyrochlore.** Si l'une des deux classes chimiques est
  vide (par exemple cinq métaux 3d classés pérovskite), les sites sont attribués par
  rayon décroissant, comme dans le reste de la plateforme, et la prédiction le signale.
  Aucune composition de validation n'est concernée.
- **Constat, rocksalt, après lecture des résidus : aucune règle modifiée.** Les trois
  compositions contenant Sr2+, K+ ou Na+ ont la maille de la matrice sans ce cation.
  Le modèle n'est pas corrigé pour autant : la plateforme signale tout rocksalt
  contenant un cation de rayon VI supérieur à 1,0 Å. Ce seuil est un avertissement, il
  ne change aucune maille prédite.

Les règles sont enregistrées dans `data/reference/lattice_rules.json`, lu à la fois par
`src/lattice.py` et par la page.

## 5. Critères de réussite

**Étape 3, modèle géométrique.**

| | critère |
|---|---|
| C1 (porte) | Sur les références, écart absolu moyen ≤ 1,0 % pour chaque famille. Si une famille échoue, l'erreur est cherchée dans les rayons, les valences ou la relation, sur les références seulement, avant toute lecture des oxydes à haute entropie. |
| C2 | Sur la validation, écart absolu moyen ≤ 1,0 % pour chacune des cinq familles. |
| C3 | Sur la validation, écart absolu moyen global inférieur à celui du modèle géométrique actuel, et aucune famille moins bonne que lui. Le modèle actuel est mesuré par `tools/lattice_dataset.py` avant l'étape 3. |

Sont rapportés sans être des critères : le biais moyen, l'écart maximal et la position
par rapport au plancher.

Pour les spinelles, le choix entre le modèle CHGNet existant et le nouveau modèle
géométrique se fait sur les spinelles de validation. C'est une sélection de modèle,
déclarée comme telle : ces spinelles ne sont plus une validation indépendante pour le
modèle retenu.

**Étape 4, correction apprise.** C'est un régresseur des résidus du modèle de l'étape 3.

- Validation croisée groupée par source : toutes les lignes d'un même article sont
  retirées ensemble.
- La correction n'est intégrée que si les deux conditions suivantes sont remplies :
  - l'écart absolu moyen hors échantillon, sur la validation, baisse d'au moins
    0,1 point et d'au moins 15 % en relatif par rapport à l'étape 3 ;
  - aucune famille ne se dégrade de plus de 0,1 point.
- Sinon, elle n'est pas intégrée et le rapport le dit.
