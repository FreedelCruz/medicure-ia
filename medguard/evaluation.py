"""Métriques d'évaluation utilisées par les notebooks 02 à 05 (numpy + scipy)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata


def auroc(y, scores) -> float:
    """AUC-ROC par les rangs (gère les ex aequo). NaN si une seule classe est présente."""
    y, s = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sensitivity_specificity(y, scores, threshold: float) -> tuple[float, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y,
    scores,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """(estimation, borne basse, borne haute) par bootstrap. Ignore les rééchantillons dégénérés."""
    y, s = np.asarray(y), np.asarray(scores)
    rng = np.random.default_rng(seed)
    point = metric(y, s)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def youden_threshold(y, scores) -> float:
    """Seuil maximisant sensibilité + spécificité - 1."""
    y, s = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    best_t, best_j = 0.5, -1.0
    for t in np.unique(s):
        sens, spec = sensitivity_specificity(y, s, t)
        if np.isnan(sens) or np.isnan(spec):
            continue
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def threshold_for_sensitivity(y, scores, target_sensitivity: float = 0.95) -> float:
    """Plus grand seuil garantissant au moins la sensibilité visée (utile pour les cas prioritaires)."""
    y, s = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    best = float(s.min())
    for t in np.sort(np.unique(s)):
        sens, _ = sensitivity_specificity(y, s, t)
        if not np.isnan(sens) and sens >= target_sensitivity:
            best = float(t)
    return best


# ------------------------------------------------------------------ prédiction sélective


def risk_coverage_curve(correct, reliability) -> dict[str, np.ndarray]:
    """Courbe risque-couverture.

    On trie les cas par fiabilité décroissante. À la couverture c (part de cas conservés),
    le risque est le taux d'erreur parmi les cas conservés. Si l'indice de fiabilité est
    utile, le risque BAISSE quand on accepte de refuser davantage de cas.
    """
    correct = np.asarray(correct).astype(float)
    rel = np.asarray(reliability, dtype=float)
    order = np.argsort(-rel, kind="stable")
    err = 1.0 - correct[order]
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return {"coverage": coverage, "risk": risk, "reliability_sorted": rel[order]}


def aurc(correct, reliability) -> float:
    """Aire sous la courbe risque-couverture (plus petit = mieux)."""
    curve = risk_coverage_curve(correct, reliability)
    return float(np.mean(curve["risk"]))


def selective_metrics(correct, reliability, threshold: float) -> dict[str, float]:
    """Couverture et exactitude si l'on refuse tous les cas dont la fiabilité < threshold."""
    correct = np.asarray(correct).astype(float)
    rel = np.asarray(reliability, dtype=float)
    kept = rel >= threshold
    return {
        "threshold": float(threshold),
        "coverage": float(kept.mean()),
        "accuracy_all": float(correct.mean()),
        "accuracy_kept": float(correct[kept].mean()) if kept.any() else float("nan"),
        "n_refused": int((~kept).sum()),
    }


def threshold_for_coverage(reliability, target_coverage: float) -> float:
    """Seuil de fiabilité qui conserve (au moins) la couverture visée."""
    rel = np.sort(np.asarray(reliability, dtype=float))
    k = int(np.floor((1.0 - target_coverage) * len(rel)))
    k = int(np.clip(k, 0, len(rel) - 1))
    return float(rel[k])
