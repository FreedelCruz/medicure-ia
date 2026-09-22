"""Composant 1 — Image Quality Assessment.

Décide si l'image est exploitable AVANT toute analyse. Chaque métrique est convertie en
sous-score 0-1 par une rampe linéaire (0 = "mauvais", 1 = "bon"). Un sous-score à 0 est
"fatal" : l'image est rejetée. Un sous-score < 0,5 produit un simple avertissement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .config import QualityThresholds
from .imaging import to_gray

_ANALYSIS_SIZE = 512


@dataclass
class QualityReport:
    usable: bool
    score: float  # 0-100
    metrics: dict[str, float] = field(default_factory=dict)
    subscores: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)  # messages lisibles (français)
    fatal_issues: list[str] = field(default_factory=list)


def _ramp(value: float, bad: float, good: float) -> float:
    """0 quand value == bad, 1 quand value == good, linéaire entre les deux (bornée)."""
    if good == bad:
        return 1.0 if value >= good else 0.0
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def compute_metrics(img: Image.Image) -> dict[str, float]:
    gray = to_gray(img)
    h, w = gray.shape
    small = cv2.resize(gray, (_ANALYSIS_SIZE, _ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    p1, p99 = np.percentile(gray, [1, 99])
    return {
        "width": float(w),
        "height": float(h),
        "min_side": float(min(h, w)),
        "blur_laplacian_var": lap_var,
        "brightness_mean": float(gray.mean()),
        "brightness_dev": float(abs(gray.mean() - 128.0)),
        "contrast_range": float(p99 - p1),
        "overexposed_fraction": float((gray >= 250).mean()),
    }


def assess_quality(img: Image.Image, thresholds: QualityThresholds | None = None) -> QualityReport:
    t = thresholds or QualityThresholds()
    m = compute_metrics(img)

    sub = {
        "blur": _ramp(m["blur_laplacian_var"], *t.blur),
        "contrast": _ramp(m["contrast_range"], *t.contrast),
        "brightness": _ramp(m["brightness_dev"], *t.brightness_dev),
        "resolution": _ramp(m["min_side"], *t.resolution),
        "exposure": _ramp(m["overexposed_fraction"], *t.overexposed_fraction),
    }

    too_dark = m["brightness_mean"] < 128.0
    messages = {
        "blur": "Image trop floue pour une analyse fiable.",
        "contrast": "Contraste insuffisant.",
        "brightness": "Image trop sombre." if too_dark else "Image surexposée (trop claire).",
        "resolution": "Résolution trop faible.",
        "exposure": "Zones surexposées trop étendues.",
    }

    fatal, warnings = [], []
    for key, value in sub.items():
        if value <= 0.0:
            fatal.append(f"⚠️ {messages[key]}")
        elif value < 0.5:
            warnings.append(f"ℹ️ {messages[key]} (limite, à surveiller)")

    # Score global : moyenne pondérée par le pire critère (une seule faiblesse suffit à gêner).
    values = np.array(list(sub.values()))
    score = 100.0 * float(0.5 * values.min() + 0.5 * values.mean())

    return QualityReport(
        usable=not fatal,
        score=round(score, 1),
        metrics={k: round(v, 3) for k, v in m.items()},
        subscores={k: round(v, 3) for k, v in sub.items()},
        issues=fatal + warnings,
        fatal_issues=fatal,
    )
