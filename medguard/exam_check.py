"""Composant 2 — Exam Classification (version heuristique v0).

Vérifie que l'image ressemble à une radiographie thoracique : image quasi monochrome,
proportions plausibles, symétrie gauche/droite. C'est un GARDE-FOU grossier, pas un
classifieur : à remplacer par un petit modèle "radio thoracique vs autre" entraîné sur
Kaggle (voir notebooks/01_eda.ipynb) et branché via ``set_classifier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from .imaging import to_gray, to_rgb_array


@dataclass
class ExamCheck:
    is_chest_xray: bool
    confidence: float  # 0-1
    reasons: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


_classifier: Optional[Callable[[Image.Image], float]] = None


def set_classifier(fn: Optional[Callable[[Image.Image], float]]) -> None:
    """Branche un vrai classifieur (image -> probabilité d'être une radio thoracique)."""
    global _classifier
    _classifier = fn


def _colorfulness(img: Image.Image) -> float:
    rgb = to_rgb_array(img).astype(np.float32)
    rg = np.abs(rgb[..., 0] - rgb[..., 1])
    yb = np.abs(0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2])
    return float(rg.mean() + yb.mean())


def _lr_symmetry(gray: np.ndarray) -> float:
    """Corrélation entre la moitié gauche et la moitié droite retournée (basse résolution)."""
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    left, right = small[:, :32], small[:, 32:][:, ::-1]
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < 1e-6:
        return 0.0
    return float((left * right).sum() / denom)


def check_exam_type(img: Image.Image, min_confidence: float = 0.55) -> ExamCheck:
    gray = to_gray(img)
    h, w = gray.shape
    color = _colorfulness(img)
    ratio = w / h
    sym = _lr_symmetry(gray)

    s_color = float(np.clip(1.0 - color / 40.0, 0.0, 1.0))  # photo couleur -> 0
    s_ratio = 1.0 if 0.6 <= ratio <= 1.6 else 0.0
    s_sym = float(np.clip((sym - 0.10) / 0.50, 0.0, 1.0))  # symétrie faible -> 0

    confidence = 0.4 * s_color + 0.2 * s_ratio + 0.4 * s_sym
    if _classifier is not None:
        confidence = 0.5 * confidence + 0.5 * float(_classifier(img))

    reasons: list[str] = []
    if s_color < 0.5:
        reasons.append(
            "L'image est en couleur : une radiographie est attendue en niveaux de gris."
        )
    if s_ratio == 0.0:
        reasons.append("Proportions inhabituelles pour une radiographie thoracique.")
    if s_sym < 0.3:
        reasons.append(
            "Structure peu symétrique : ne ressemble pas à une radiographie thoracique de face."
        )

    # Garde-fous durs : une image très colorée, aux proportions absurdes ou sans symétrie est refusée
    # même si la moyenne pondérée semble acceptable.
    ok = (
        confidence >= min_confidence and s_color >= 0.3 and s_ratio > 0 and s_sym >= 0.3
    )
    return ExamCheck(
        is_chest_xray=ok,
        confidence=round(confidence, 3),
        reasons=[] if ok else reasons,
        details={
            "colorfulness": round(color, 2),
            "aspect_ratio": round(ratio, 3),
            "lr_symmetry": round(sym, 3),
        },
    )
