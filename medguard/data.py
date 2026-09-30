"""Chargement des datasets publics et inférence par lots (utilisé par les notebooks).

* NIH ChestX-ray14 : ``Data_Entry_2017.csv`` + dossiers ``images_*/images/*.png``.
  Étiquettes disponibles pour nos cibles : Effusion, Cardiomegaly (PAS Lung Opacity).
* CheXpert : ``train.csv`` / ``valid.csv`` avec colonne ``Path``. Valeurs : 1 (positif),
  0 (négatif), -1 (INCERTAIN), vide (non mentionné).

Les chemins Kaggle varient selon la version du dataset attaché : adaptez ``root`` si besoin.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


import numpy as np
import pandas as pd

from .imaging import load_image, to_gray

# nom TorchXRayVision -> colonne CheXpert
CHEXPERT_COLUMNS = {"Effusion": "Pleural Effusion", "Cardiomegaly": "Cardiomegaly", "Lung Opacity": "Lung Opacity"}
NIH_TARGETS = ("Effusion", "Cardiomegaly")


def load_nih(root: str | Path, targets: Sequence[str] = NIH_TARGETS) -> pd.DataFrame:
    """Une ligne par image : image_id, patient_id, path, age, sex, view + une colonne 0/1 par cible."""
    root = Path(root)
    csv_path = next(iter(root.rglob("Data_Entry_2017*.csv")), None)
    if csv_path is None:
        raise FileNotFoundError(f"Data_Entry_2017.csv introuvable sous {root}")
    df = pd.read_csv(csv_path)

    index = {p.name: p for p in root.rglob("*.png")}
    out = pd.DataFrame(
        {
            "image_id": df["Image Index"],
            "patient_id": df["Patient ID"].astype(str),
            "age": pd.to_numeric(df.get("Patient Age"), errors="coerce"),
            "sex": df.get("Patient Gender"),
            "view": df.get("View Position"),
        }
    )
    out["path"] = out["image_id"].map(lambda n: str(index[n]) if n in index else None)
    labels = df["Finding Labels"].fillna("").str.split("|")
    for t in targets:
        out[t] = labels.map(lambda ls, t=t: float(t in ls))
    return out.dropna(subset=["path"]).reset_index(drop=True)


def load_chexpert(root: str | Path, csv_name: str = "valid.csv", targets: Sequence[str] = tuple(CHEXPERT_COLUMNS)) -> pd.DataFrame:
    """Radiographies de FACE uniquement. Valeurs cibles : 1, 0, -1 (incertain) ou NaN."""
    root = Path(root)
    csv_path = next(iter(root.rglob(csv_name)), None)
    if csv_path is None:
        raise FileNotFoundError(f"{csv_name} introuvable sous {root}")
    df = pd.read_csv(csv_path)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"]

    def resolve(rel: str) -> str | None:
        for cand in (root / rel, root.parent / rel, root / rel.split("/", 1)[-1], csv_path.parent / rel.split("/", 1)[-1]):
            if cand.exists():
                return str(cand)
        return None

    out = pd.DataFrame(
        {
            "image_id": df["Path"],
            "patient_id": df["Path"].map(lambda p: (re.search(r"patient(\d+)", p) or [None, p])[1]),
            "age": pd.to_numeric(df.get("Age"), errors="coerce"),
            "sex": df.get("Sex"),
            "view": df.get("AP/PA"),
            "path": df["Path"].map(resolve),
        }
    )
    for t in targets:
        out[t] = pd.to_numeric(df[CHEXPERT_COLUMNS[t]], errors="coerce")
    return out.dropna(subset=["path"]).reset_index(drop=True)


def binarize(series: pd.Series, uncertain: str = "drop") -> pd.Series:
    """Convertit une colonne CheXpert en 0/1. ``uncertain`` : 'drop' (NaN), 'positive' ou 'negative'.

    'drop' est le choix le plus honnête pour ÉVALUER (on ne juge pas le modèle sur des étiquettes
    que les radiologues eux-mêmes jugeaient incertaines) ; les cas incertains servent en revanche à
    illustrer le refus de conclure.
    """
    s = series.copy().astype(float)
    if uncertain == "positive":
        s[s == -1] = 1.0
    elif uncertain == "negative":
        s[s == -1] = 0.0
    elif uncertain == "drop":
        s[s == -1] = np.nan
    else:
        raise ValueError(uncertain)
    return s


def sample_by_patient(df: pd.DataFrame, n_max: int, seed: int = 42) -> pd.DataFrame:
    """Sous-échantillon d'au plus ~n_max images en gardant les patients entiers."""
    if len(df) <= n_max:
        return df
    rng = np.random.default_rng(seed)
    patients = np.asarray(df["patient_id"].unique(), dtype=object)
    rng.shuffle(patients)
    keep, total = [], 0
    counts = df["patient_id"].value_counts()
    for p in patients:
        keep.append(p)
        total += int(counts[p])
        if total >= n_max:
            break
    return df[df["patient_id"].isin(keep)].reset_index(drop=True)


def predict_paths(predictor, paths: Sequence[str], batch_size: int = 32, progress: bool = True) -> np.ndarray:
    """Scores bruts (n, T) d'un ``Predictor`` sur une liste de fichiers image."""
    out = []
    rng = range(0, len(paths), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            rng = tqdm(rng, desc="inférence")
        except ImportError:
            pass
    for i in rng:
        grays = [to_gray(load_image(p)) for p in paths[i : i + batch_size]]
        out.append(np.asarray(predictor.predict(grays)))
    return np.concatenate(out) if out else np.zeros((0, len(predictor.targets)))


def load_dataset(dataset: str, root: str | Path, n_max: int = 6000, seed: int = 42, chexpert_csv: str = "valid.csv"):
    """Point d'entrée unique des notebooks. Retourne (DataFrame, cibles évaluables).

    Les colonnes cibles valent 0, 1 ou NaN (étiquette incertaine ou absente : exclue de l'évaluation).
    """
    from .config import TARGETS

    dataset = dataset.lower()
    if dataset == "nih":
        df, eval_targets = load_nih(root), list(NIH_TARGETS)
    elif dataset == "chexpert":
        df, eval_targets = load_chexpert(root, chexpert_csv), list(TARGETS)
        for t in eval_targets:
            df[t] = binarize(df[t], "drop")
    else:
        raise ValueError(f"Dataset inconnu : {dataset!r} (attendu : 'nih' ou 'chexpert')")
    return sample_by_patient(df, n_max, seed), eval_targets
