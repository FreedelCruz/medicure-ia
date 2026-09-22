"""Retrouver un fichier produit par un autre notebook, où qu'on soit (Kaggle ou local).

Ordre de recherche :
1. les dossiers passés en argument (ex. le dossier de travail courant) ;
2. les Kaggle Datasets attachés : ``/kaggle/input/*/`` (recommandé : publiez ``artifacts/`` et
   ``data/splits/`` comme un Dataset privé "medguard-artifacts") ;
3. les dossiers du dépôt : ``artifacts/`` et ``data/splits/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import ARTIFACTS_DIR, ROOT_DIR


def candidate_dirs(extra: Iterable[str | Path] = ()) -> list[Path]:
    dirs = [Path(d) for d in extra]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        dirs += [d for d in sorted(kaggle_input.iterdir()) if d.is_dir()]
    dirs += [ARTIFACTS_DIR, ROOT_DIR / "data" / "splits"]
    return dirs


def find_file(name: str, extra_dirs: Iterable[str | Path] = ()) -> Path:
    for d in candidate_dirs(extra_dirs):
        direct = d / name
        if direct.exists():
            return direct
        nested = next(iter(d.rglob(name)), None) if d.exists() and d.is_dir() and "kaggle/input" in str(d) else None
        if nested is not None:
            return nested
    searched = ", ".join(str(d) for d in candidate_dirs(extra_dirs))
    raise FileNotFoundError(f"'{name}' introuvable. Dossiers examinés : {searched}")
