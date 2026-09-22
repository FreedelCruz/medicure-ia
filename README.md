
---
title: Medicure AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---


# Médicure AI — MedGuard Engine

> **L'IA observe. Le professionnel de santé décide.**

Aide à la lecture de radiographies thoraciques (épanchement pleural, cardiomégalie, opacité pulmonaire) dont
la particularité est de **savoir refuser** : contrôle qualité avant analyse, scores calibrés, indice de fiabilité,
« refus de conclure » quand la confiance est insuffisante, explications Grad-CAM, journal d'audit.

⚠️ Prototype de hackathon. **Ce n'est pas un dispositif médical** et il ne remplace ni un médecin ni un radiologue.

## Démarrage rapide (sans torch, sans données)

```bash
pip install -e ".[app,dev]"        # ou : make install (inclut torch + torchxrayvision)
python -m pytest                   # 60+ tests, quelques secondes
python -m app.gradio_app --mock    # interface en MODE SIMULATION (scores fictifs)
make notebooks-smoke               # exécute les 5 notebooks sur un faux dataset
```

Avec le vrai modèle : `pip install -e ".[model,app]"` puis `python -m app.gradio_app`.

## Structure

```
medguard/            cœur du système (importé par l'app ET par les notebooks)
  quality.py           contrôle qualité : flou, contraste, luminosité, résolution
  exam_check.py        est-ce une radio thoracique ? (heuristique v0)
  model.py             TorchXRayVision + Grad-CAM  (seul module qui importe torch)
  calibration.py       ECE, temperature / Platt scaling
  reliability.py       indice de fiabilité, refus de conclure, priorité
  explain.py           zone + phrase prudente + superposition
  pipeline.py          orchestration de bout en bout
  report.py, audit.py  rapport HTML, journal d'audit (JSONL)
  evaluation.py        AUROC, IC bootstrap, risque–couverture
  splits.py, data.py   découpage par patient, chargeurs NIH / CheXpert
  mock.py, synthetic.py  prédicteur SIMULÉ et images synthétiques (tests, développement)
app/                 interface Gradio + API FastAPI optionnelle
notebooks/           01 EDA · 02 baseline · 03 calibration · 04 risque–couverture · 05 robustesse
artifacts/           JSON produits par les notebooks (calibration, seuils) — à commiter
data/splits/         découpages par patient — à commiter
docs/                logique de décision, guide Kaggle, script de démo, cahier des charges
tests/, scripts/
```

## Flux de travail

1. **VS Code** : code dans `medguard/`, tests, application. Poussez sur GitHub.
2. **Kaggle** : les notebooks 01 → 05 (voir `docs/KAGGLE.md`) produisent des artefacts.
3. **Retour** : commitez `data/splits/splits.json`, `artifacts/calibration.json`, `operating_points.json`, `thresholds.json`.
4. L'application charge ces artefacts au démarrage. Tant que `calibration.json` est absent, elle affiche des
   « scores bruts non calibrés » (jamais de faux pourcentage).

## Mettre en ligne

Voir `docs/DEPLOY.md` — déploiement recommandé sur Hugging Face Spaces (SDK Docker, gratuit en
CPU). Le `Dockerfile`, `app.py` (point d'entrée conteneur) et `requirements-space.txt` sont prêts
à l'emploi.

## Ce qui a été vérifié, et ce qui ne l'a pas été

**Exécuté et vérifié (suite de tests + exécution des notebooks) :** contrôle qualité, détecteur d'examen, calibration,
indice de fiabilité et logique de priorité/refus, découpage par patient, chargeurs de données, pipeline complet avec
prédicteur simulé, audit, rapport, interface Gradio (appelée via HTTP), API FastAPI, et les 5 notebooks de bout en bout
sur un faux dataset avec le prédicteur simulé.

**NON vérifié — à faire dès que vous avez torch :**
- `medguard/model.py` (`XRVPredictor`, Grad-CAM) : écrit d'après le code source de TorchXRayVision 1.5.4, mais jamais exécuté
  ici (torch non installable dans mon environnement). Lancez `pytest tests/test_model_torch.py` : il utilise des poids aléatoires,
  sans téléchargement, pour valider la plomberie.
- Les notebooks sur **vraies données** : chemins Kaggle des datasets, temps de calcul, valeurs réelles.
- **Tous les seuils** (`config.py`) sont des valeurs de départ. Ceux du contrôle qualité et du refus doivent être recalés
  (notebooks 01 et 04). Les images synthétiques des tests n'ont pas les propriétés de vraies radios.
- Aucun chiffre de performance n'est fourni : les sorties des notebooks exécutés ici sont **fictives**.

## Premières actions

- [ ] Remplacer `VOTRE_COMPTE` dans la cellule d'initialisation des notebooks
- [ ] `pytest tests/test_model_torch.py` avec torch installé
- [ ] Demander l'accès à CheXpert (accord d'usage recherche) si vous voulez l'utiliser
- [ ] Notebook 01 sur Kaggle → commiter `splits.json`
- [ ] Notebooks 02 → 04 → commiter les artefacts, lire la courbe risque–couverture **avant** de rédiger le discours
- [ ] Choisir et vérifier vos 5 images de démo avec le vrai modèle (`demo_assets/README.md`)
