"""Découpage reproductible des données PAR PATIENT.

Un même patient peut avoir plusieurs radiographies. S'il apparaît dans deux splits, le
modèle est évalué sur des images qu'il a "déjà vues" (à un autre moment) et les résultats
sont trop optimistes. On découpe donc par groupe (patient), une seule fois, et on commite
le fichier de splits pour que tout le monde évalue sur les mêmes images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_FRACTIONS = {"dev": 0.20, "calibration": 0.30, "test": 0.50}


def group_split(
    groups: Sequence,
    fractions: dict[str, float] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Retourne, pour chaque ligne, le nom de son split. Un groupe n'est jamais coupé en deux."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Les fractions doivent sommer à 1 (reçu {total}).")

    groups_arr = np.asarray(groups)
    unique = np.unique(groups_arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    names = list(fractions)
    bounds = np.cumsum([fractions[n] for n in names]) * len(unique)
    group_to_split: dict = {}
    for i, g in enumerate(unique):
        k = int(np.searchsorted(bounds, i + 1e-9, side="right"))
        group_to_split[g] = names[min(k, len(names) - 1)]

    return np.array([group_to_split[g] for g in groups_arr])


def save_splits(path: str | Path, ids: Sequence, split_names: Sequence, meta: dict | None = None) -> Path:
    """Sauvegarde {split: [identifiants]} en JSON (à commiter dans data/splits/)."""
    out: dict = {"meta": meta or {}, "splits": {}}
    ids, split_names = np.asarray(ids), np.asarray(split_names)
    for name in np.unique(split_names):
        out["splits"][str(name)] = [str(i) for i in ids[split_names == name]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["splits"]
