# Travailler avec Kaggle

Principe : **Kaggle calcule (GPU, données), VS Code construit (code, appli, tests)**. Ils échangent
des **artefacts** (JSON), jamais des cellules copiées-collées.

## 1. Mettre le code à disposition de Kaggle

**Option A — GitHub public** (le plus simple). Dans la cellule d'initialisation de chaque notebook,
remplacez `VOTRE_COMPTE` par votre identifiant :
`%pip install -q git+https://github.com/VOTRE_COMPTE/medicure-ai.git torchxrayvision`
Il faut **activer Internet** dans les réglages du notebook Kaggle.

**Option B — dépôt privé / hors ligne.** Zippez le dossier `medguard/` + `pyproject.toml`, publiez-le comme
Dataset Kaggle privé « medguard-src », attachez-le, puis :
`%pip install -q /kaggle/input/medguard-src` (adaptez le chemin au contenu du Dataset).

## 2. Attacher les données (à vérifier sur la plateforme)

Les noms de Datasets et leur arborescence changent selon leurs auteurs et leurs versions. Points de départ
probables, **non vérifiés ici** :

| Données | Recherche Kaggle | Attendu |
|---|---|---|
| NIH ChestX-ray14 | « NIH Chest X-rays » (`nih-chest-xrays/data`) | `Data_Entry_2017.csv` + `images_*/images/*.png` |
| CheXpert (version réduite) | « CheXpert » (ex. `ashery/chexpert`) | `train.csv`, `valid.csv`, images `.jpg` |

Dans un notebook : `!ls /kaggle/input` puis `!ls /kaggle/input/<dataset>` pour trouver la racine, et
réglez `DATA_ROOT` dans la cellule de configuration. Vérifiez la **licence** de chaque miroir ; préférez la source officielle.

## 3. Ordre d'exécution et ce qu'on récupère

| # | Notebook | Produit | À faire du résultat |
|---|---|---|---|
| 01 | `01_eda` | `splits.json`, figures | copier `splits.json` dans `data/splits/`, **commiter** |
| 02 | `02_baseline_eval` | `artifacts/raw_scores.npz` | garder pour le 03 (pas dans Git) |
| 03 | `03_calibration` | `calibration.json`, `operating_points.json` | copier dans `artifacts/`, **commiter** |
| 04 | `04_risk_coverage` | `thresholds.json`, `risk_coverage.png` | copier dans `artifacts/`, **commiter** |
| 05 | `05_robustness` | `robustness.csv`, figure | pour les slides |

**Enchaîner les notebooks** (chaque session Kaggle est indépendante) : après chaque notebook, publiez le
contenu de `/kaggle/working/artifacts` (et `splits.json`) comme **Dataset privé « medguard-artifacts »**, puis
attachez-le aux notebooks suivants. `medguard.paths.find_file` le retrouve automatiquement dans `/kaggle/input/*/`.

Avec l'API en ligne de commande Kaggle, depuis votre machine :

```bash
kaggle kernels output <utilisateur>/<notebook> -p ./from_kaggle      # récupérer les sorties
# puis, pour publier une nouvelle version du Dataset d'artefacts :
kaggle datasets version -p ./artifacts_upload -m "calibration v2"    # exige un dataset-metadata.json
```

## 4. Fuite de données : quels poids évaluer sur quel dataset

| Poids (`MODEL_WEIGHTS`) | Entraînés sur | Évaluer sur |
|---|---|---|
| `densenet121-res224-chex` (défaut) | CheXpert | **NIH** (externe) |
| `densenet121-res224-nih` | NIH | **CheXpert** (externe) |
| `densenet121-res224-all` | la plupart des datasets publics | aucun n'est réellement externe |

Avec `-chex`, la cible *Lung Opacity* n'a pas d'étiquette dans NIH : elle ne sera ni calibrée ni évaluée (l'application
l'affichera « score brut, non calibré »). Pour la calibrer, il faut un jeu étiqueté qui la contient, sans fuite.

## 5. Poids hors ligne

TorchXRayVision télécharge ses poids au premier lancement. Sans Internet : téléchargez-les ailleurs (dossier `~/.torchxrayvision/`),
publiez-les comme Dataset, attachez-le et définissez `MEDGUARD_WEIGHTS_CACHE=/kaggle/input/<dataset-poids>`.

## 6. Quotas

Les heures GPU et la durée maximale des sessions varient : **vérifiez-les sur votre compte** avant de planifier. Réduisez `MEDGUARD_NMAX`,
`MEDGUARD_NEVAL` et `MEDGUARD_NROBUST` pour un premier passage rapide. Plan B : Google Colab.

## 7. Tester sans rien télécharger

`make notebooks-smoke` fabrique un faux dataset et exécute les 5 notebooks avec le prédicteur simulé (`MEDGUARD_MOCK=1`).
À lancer avant Kaggle pour attraper les fautes de chemins et de formats.
