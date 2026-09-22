"""Calibration des probabilités (numpy uniquement).

Un score n'a de valeur que s'il est calibré : quand le système annonce 0,9, il doit avoir
raison environ 9 fois sur 10. Ce module fournit :

* ``expected_calibration_error`` et ``reliability_curve`` pour MESURER la calibration ;
* ``fit_temperature`` (1 paramètre) et ``fit_platt`` (2 paramètres) pour la CORRIGER ;
* ``Calibrator`` : un calibrateur par pathologie, sauvegardé dans ``artifacts/calibration.json``.

Règle d'or : ajuster sur le split "calibration", mesurer sur le split "test". Jamais les deux
sur les mêmes images.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-6


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ----------------------------------------------------------------------------- mesures


def brier_score(probs, y) -> float:
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((probs - y) ** 2))


def reliability_curve(probs, y, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Courbe de fiabilité (probabilité prédite vs fréquence observée) sur la classe positive."""
    probs, y = np.asarray(probs, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        counts.append(int(mask.sum()))
        mean_pred.append(float(probs[mask].mean()) if mask.any() else np.nan)
        frac_pos.append(float(y[mask].mean()) if mask.any() else np.nan)
    return {
        "bin_edges": edges,
        "mean_predicted": np.array(mean_pred),
        "fraction_positive": np.array(frac_pos),
        "counts": np.array(counts),
    }


def expected_calibration_error(probs, y, n_bins: int = 10) -> float:
    """ECE sur la classe positive : somme pondérée des écarts |fréquence observée - prédite|."""
    curve = reliability_curve(probs, y, n_bins)
    counts = curve["counts"].astype(float)
    total = counts.sum()
    if total == 0:
        return float("nan")
    gaps = np.abs(np.nan_to_num(curve["fraction_positive"]) - np.nan_to_num(curve["mean_predicted"]))
    return float(np.sum(counts * gaps) / total)


# ----------------------------------------------------------------------------- ajustement


def _nll(z, y) -> float:
    p = np.clip(sigmoid(z), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits, y, t_min: float = 0.05, t_max: float = 20.0, n: int = 800) -> float:
    """Température T minimisant la log-vraisemblance négative de sigmoid(logit / T)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=float)
    grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), n))
    losses = [_nll(logits / t, y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def fit_platt(logits, y, l2: float = 1e-3) -> tuple[float, float]:
    """Régression logistique 1D : p = sigmoid(a * logit + b), minimisée par L-BFGS.

    (Un Newton "nu" diverge dès que le modèle est très sur-confiant : la hessienne est alors
    minuscule et le premier pas part très loin. L-BFGS avec gradient exact est robuste.)
    Une petite régularisation L2 tire (a, b) vers (1, 0), c'est-à-dire vers "pas de correction".
    """
    from scipy.optimize import minimize

    x, y = np.asarray(logits, dtype=float), np.asarray(y, dtype=float)

    def objective(theta):
        a, b = theta
        z = a * x + b
        p = np.clip(sigmoid(z), EPS, 1 - EPS)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * ((a - 1.0) ** 2 + b**2)
        resid = sigmoid(z) - y
        grad = np.array([np.mean(resid * x) + l2 * (a - 1.0), np.mean(resid) + l2 * b])
        return loss, grad

    res = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


# ----------------------------------------------------------------------------- calibrateur


@dataclass
class TargetCalibration:
    method: str = "identity"  # "identity" | "temperature" | "platt"
    a: float = 1.0  # 1/T pour temperature ; pente pour platt
    b: float = 0.0

    def apply(self, probs):
        if self.method == "identity":
            return np.asarray(probs, dtype=float)
        return sigmoid(self.a * logit(probs) + self.b)


@dataclass
class Calibrator:
    """Un calibrateur par pathologie. Sans ajustement, il laisse les scores inchangés."""

    per_target: dict[str, TargetCalibration] = field(default_factory=dict)
    fitted_on: str = ""  # description libre du split / dataset utilisé

    def fit(self, raw_probs: dict[str, np.ndarray], labels: dict[str, np.ndarray], method: str = "platt"):
        for name, p in raw_probs.items():
            y = np.asarray(labels[name], dtype=float)
            z = logit(p)
            if method == "temperature":
                self.per_target[name] = TargetCalibration("temperature", 1.0 / fit_temperature(z, y), 0.0)
            elif method == "platt":
                a, b = fit_platt(z, y)
                self.per_target[name] = TargetCalibration("platt", a, b)
            else:
                raise ValueError(f"Méthode inconnue : {method}")
        return self

    def calibrate(self, name: str, probs):
        return self.per_target.get(name, TargetCalibration()).apply(probs)

    def is_fitted(self, name: str) -> bool:
        return name in self.per_target and self.per_target[name].method != "identity"

    # --- persistance
    def to_dict(self) -> dict:
        return {
            "fitted_on": self.fitted_on,
            "targets": {k: vars(v) for k, v in self.per_target.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        cal = cls(fitted_on=data.get("fitted_on", ""))
        for k, v in data.get("targets", {}).items():
            cal.per_target[k] = TargetCalibration(**v)
        return cal

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
