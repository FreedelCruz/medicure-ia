#!/usr/bin/env bash
# Exécute les 5 notebooks de bout en bout sur un FAUX dataset avec le prédicteur SIMULÉ.
# Vérifie que la plomberie fonctionne (chemins, formats, sorties) avant de lancer sur Kaggle.
# Les chiffres obtenus n'ont AUCUNE valeur médicale.
set -euo pipefail
cd "$(dirname "$0")/.."

FAKE=/tmp/medguard_fake_nih
WORK=/tmp/medguard_smoke
rm -rf "$WORK"; mkdir -p "$WORK/executed"

python scripts/make_fake_dataset.py --out "$FAKE" --patients 150

export MEDGUARD_MOCK=1 MEDGUARD_DATA="$FAKE" MEDGUARD_WORK="$WORK" MEDGUARD_NEVAL=200 MEDGUARD_NROBUST=40
cd notebooks
for nb in 01_eda 02_baseline_eval 03_calibration 04_risk_coverage 05_robustness; do
  echo "=== $nb"
  python -m jupyter nbconvert --to notebook --execute "$nb.ipynb" --output-dir "$WORK/executed"
done
echo "OK — notebooks exécutés dans $WORK/executed"
