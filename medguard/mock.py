"""Prédicteur SIMULÉ — pour développer l'interface et les tests sans torch ni poids.

⚠️  Les scores produits ici sont FICTIFS. Ils ne reposent sur aucune analyse médicale.
L'interface affiche une bannière "MODE SIMULATION" tant que ``info['simulation']`` est vrai.
Ne jamais utiliser ce prédicteur pour la démo finale devant le jury.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from .config import INPUT_SIZE, TARGETS
from .preprocess import center_crop_square


class MockPredictor:
    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        jitter: float = 0.0,
        targets: Sequence[str] | None = None,
        seed: int = 0,
    ):
        """
        scores : scores constants par pathologie (sinon dérivés de statistiques d'image).
        jitter : écart-type d'un bruit gaussien ajouté à chaque prédiction, pour simuler
                 un modèle instable sous perturbations.
        """
        self.targets = list(targets or TARGETS)
        self._fixed = dict(scores) if scores else None
        self._jitter = float(jitter)
        self._rng = np.random.default_rng(seed)
        self.info = {
            "backend": "mock",
            "simulation": True,
            "weights": "aucun (simulation)",
            "note": "Scores fictifs — aucune valeur médicale.",
        }

    def _derived(self, gray: np.ndarray) -> np.ndarray:
        """Scores continus dérivés de la luminosité de quelques zones (stables sous petites perturbations)."""
        small = cv2.resize(center_crop_square(gray), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        g_mean, g_std = small.mean(), small.std() + 1e-6
        zones = {
            "Effusion": small[20:, :].mean(),  # bas du thorax
            "Cardiomegaly": small[12:26, 8:24].mean(),  # centre
            "Lung Opacity": small[6:20, :].mean(),  # champs pulmonaires
        }
        out = []
        for t in self.targets:
            z = (zones.get(t, g_mean) - g_mean) / g_std
            out.append(1.0 / (1.0 + np.exp(-2.0 * z)))
        return np.array(out)

    def predict(self, grays: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for g in grays:
            if self._fixed is not None:
                row = np.array([self._fixed.get(t, 0.1) for t in self.targets], dtype=float)
            else:
                row = self._derived(g)
            if self._jitter > 0:
                row = row + self._rng.normal(0.0, self._jitter, row.shape)
            rows.append(np.clip(row, 0.0, 1.0))
        return np.stack(rows)

    def explain(self, gray: np.ndarray, target: str) -> Optional[np.ndarray]:
        """Carte fictive : une tache gaussienne dont la position dépend de la pathologie."""
        centers = {"Effusion": (0.75, 0.30), "Cardiomegaly": (0.55, 0.55), "Lung Opacity": (0.45, 0.72)}
        cy, cx = centers.get(target, (0.5, 0.5))
        yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE] / float(INPUT_SIZE)
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 0.12**2)))
        return (blob / blob.max()).astype(np.float32)
