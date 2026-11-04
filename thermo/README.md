# thermo/ — descripteurs thermodynamiques calculés (étapes 1 et 2)

Le classifieur de `tools/` répond à « quelle famille la littérature rapporte-t-elle pour
ces cations ». Ce dossier calcule, à partir de la seule composition, des grandeurs
physiques qui répondent aux questions du laboratoire : la composition peut-elle être
monophasée ? sous quelle structure dans l'air ? avec quel paramètre de maille ?

Le calcul utilise un potentiel interatomique universel appris, **CHGNet** (Deng et al.,
*Nat. Mach. Intell.* 2023 ; modèle pré-entraîné v0.3.0 sur les trajectoires DFT GGA/GGA+U
du Materials Project), à la place de la DFT : quelques secondes à quelques minutes par
supercellule sur un CPU au lieu de jours. Tout est confronté à des mesures avant usage.

| Script | Question | Juge |
|---|---|---|
| `run_rocksalt_benchmark.py` | enthalpie de mélange et désordre de liaison des rocksalt | les 8 compositions synthétisées par Sivak et al. (PRL 2025) |
| `run_redox_validation.py` | CHGNet sait-il l'énergie d'oxydation des cations ? | NIST-JANAF et Jacob et al. sur 5 couples redox |
| `run_oxidation_descriptor.py` | rocksalt ou spinelle dans l'air ? | les étiquettes de la base |
| `run_spinel_lattice.py` | paramètre de maille des spinelles HEO | 4 spinelles HEO mesurés, dont l'échantillon interne |
| `analyze_rocksalt_spinel.py` | le descripteur aide-t-il le classifieur ? substituts additifs pour la page | validation croisée groupée, cation retiré |
| `analyze_oer_literature.py` | les surtensions publiées peuvent-elles entraîner un prédicteur ? | corrélation intraclasse |
| `run_vacancy.py` | énergie de formation d'une lacune d'oxygène | enthalpies de réduction mesurées de 9 oxydes binaires |

Les résultats bruts sont dans `data/thermo/`, les rapports dans `QC/thermo_*.txt`.
Chaque script de calcul reprend là où il s'est arrêté : on peut l'interrompre et le
relancer, ou le découper en tranches (`run_oxidation_descriptor.py --limit 20`).

## Ordre d'exécution et durées (2 cœurs, sans GPU)

```bash
python thermo/run_rocksalt_benchmark.py          # ~40 min
python thermo/run_redox_validation.py            # ~10 min
python thermo/run_spinel_lattice.py              # ~45 min
python thermo/run_oxidation_descriptor.py --check-protocol   # ~45 min, bruit numérique
python thermo/run_oxidation_descriptor.py        # 2 à 6 min par système ; 80 des 118 calculés
python thermo/run_vacancy.py                     # ~2 h 30 (9 binaires, 14 spinelles HEO)
python thermo/analyze_rocksalt_spinel.py         # quelques minutes, sans CHGNet
python thermo/analyze_oer_literature.py
python tools/active_learning.py                  # étape 3, sans CHGNet
python tools/build_app.py                        # reporte les résultats dans la page
```

Les critères d'acceptation de chaque étape ont été fixés **avant** les calculs ; quand un
critère n'a pas été atteint, le rapport le dit et la page n'utilise pas le résultat
correspondant (voir `docs/RAPPORT_plateforme_HEO.md`, §11 à §13).

## Installation

```bash
pip install torch pymatgen ase nvidia-ml-py
pip install --no-deps chgnet==0.4.2
# Python >= 3.13 : nvidia-ml-py3, dépendance de CHGNet, ne compile pas ; une ligne suffit
python -c "import site,os; open(os.path.join(site.getsitepackages()[0],'nvidia_smi.py'),'w').write('from pynvml import *\n')"
```

Prévoir ~5 Go de mémoire par relaxation de 120 atomes ; deux cœurs suffisent.
