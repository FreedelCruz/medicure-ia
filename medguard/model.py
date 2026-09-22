"""Composants 3-5 — Prédiction (TorchXRayVision) et Grad-CAM.

Ce module est le SEUL qui importe torch, et il le fait paresseusement : tout le reste de
``medguard`` (qualité, calibration, fiabilité, rapport, tests) fonctionne sans torch.

Interface attendue par le pipeline (``Predictor``) :

* ``targets``  : liste ordonnée des pathologies ciblées ;
* ``predict``  : liste d'images (uint8, H×W) -> tableau (n, len(targets)) de scores bruts [0, 1] ;
* ``explain``  : image + pathologie -> carte de chaleur (224×224, valeurs dans [0, 1]) ou None ;
* ``info``     : dictionnaire décrivant le modèle (pour le rapport et l'audit).

Attention : avec les poids TorchXRayVision qui définissent des ``op_threshs``, les sorties sont
normalisées autour d'un point de fonctionnement (0,5 = seuil optimal du modèle). Ce ne sont PAS
des probabilités calibrées : d'où la calibration obligatoire (medguard.calibration).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import INPUT_SIZE, MODEL_WEIGHTS, TARGETS
from .preprocess import preprocess_gray


@runtime_checkable
class Predictor(Protocol):
    targets: list[str]
    info: dict

    def predict(self, grays: Sequence[np.ndarray]) -> np.ndarray: ...

    def explain(self, gray: np.ndarray, target: str) -> Optional[np.ndarray]: ...


class XRVPredictor:
    """DenseNet121 pré-entraîné de TorchXRayVision, restreint aux pathologies ciblées."""

    def __init__(
        self,
        weights: str = MODEL_WEIGHTS,
        targets: Sequence[str] | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        """``cache_dir`` (ou MEDGUARD_WEIGHTS_CACHE) : dossier où trouver/télécharger les poids.
        Utile hors ligne : pré-téléchargez-les puis attachez-les comme Dataset Kaggle."""
        import torch
        import torchxrayvision as xrv

        try:  # torch compilé pour NumPy 1.x + NumPy 2.x installé => "Numpy is not available"
            torch.from_numpy(np.zeros(1, dtype=np.float32))
        except Exception as exc:
            raise RuntimeError(
                f"PyTorch {torch.__version__} ne parvient pas à utiliser NumPy {np.__version__}. "
                'Cause fréquente : torch compilé pour NumPy 1.x. Corrigez avec : pip install "numpy<2" '
                '"opencv-python-headless<4.12" (puis relancez).'
            ) from exc

        self._torch = torch
        self._xrv = xrv
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cache_dir = cache_dir or os.environ.get("MEDGUARD_WEIGHTS_CACHE") or None
        self.model = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir).to(self.device).eval()

        self.targets = list(targets or TARGETS)
        available = list(self.model.pathologies)
        missing = [t for t in self.targets if t not in available]
        if missing:
            raise ValueError(
                f"Les poids '{weights}' ne prédisent pas : {missing}. "
                f"Sorties valides : {[p for p in available if p]}"
            )
        self._idx = {t: available.index(t) for t in self.targets}
        self.info = {
            "backend": "torchxrayvision",
            "weights": weights,
            "torchxrayvision_version": getattr(xrv, "__version__", "?"),
            "input_size": INPUT_SIZE,
            "device": str(self.device),
            "note": "Scores normalisés autour d'un point de fonctionnement, non calibrés par défaut.",
        }

    # ------------------------------------------------------------------ prédiction
    def predict(self, grays: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = np.stack([preprocess_gray(g) for g in grays])  # (n, 1, 224, 224)
        with torch.no_grad():
            out = self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
        cols = [out[:, self._idx[t]] for t in self.targets]
        return np.clip(np.stack(cols, axis=1), 0.0, 1.0)

    # ------------------------------------------------------------------ Grad-CAM
    def explain(self, gray: np.ndarray, target: str) -> Optional[np.ndarray]:
        """Grad-CAM sur la sortie de ``model.features`` (avant ReLU), classe ``target``.

        On rejoue le passage avant à la main plutôt que d'appeler ``model.forward`` : la
        version de TorchXRayVision applique un ReLU *in-place* sur la couche visée, ce qui
        rend les hooks de gradient ambigus. Le gradient est pris sur le logit brut (avant
        sigmoïde et normalisation au point de fonctionnement).
        """
        torch = self._torch
        F = torch.nn.functional
        if target not in self._idx:
            return None

        x = torch.from_numpy(preprocess_gray(gray))[None].to(self.device)
        x.requires_grad_(True)
        with torch.enable_grad():
            feats = self.model.features(x)  # (1, C, h, w)
            act = F.relu(feats)  # PAS in-place
            pooled = F.adaptive_avg_pool2d(act, (1, 1)).flatten(1)
            logit = self.model.classifier(pooled)[0, self._idx[target]]
            grads = torch.autograd.grad(logit, feats)[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        peak = float(cam.max())
        return cam / peak if peak > 1e-8 else np.zeros_like(cam)


def get_predictor(weights: str = MODEL_WEIGHTS, mock: bool | None = None) -> Predictor:
    """Vrai modèle, ou prédicteur SIMULÉ si ``mock`` est vrai (ou si MEDGUARD_MOCK=1).

    Le mode simulation sert uniquement aux tests hors ligne des notebooks et de l'interface.
    """
    if mock is None:
        mock = os.environ.get("MEDGUARD_MOCK") == "1"
    if mock:
        from .mock import MockPredictor

        return MockPredictor()
    return XRVPredictor(weights)
