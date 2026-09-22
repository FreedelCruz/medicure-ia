"""Composant 5 — Explicabilité (partie sans torch).

Transforme une carte de chaleur Grad-CAM en (1) une zone décrite en français, (2) une phrase
explicative prudente, (3) une image superposée. Le calcul Grad-CAM lui-même vit dans
``model.py`` car il dépend de l'architecture.

Limites à assumer à l'oral :
* la carte Grad-CAM du DenseNet est calculée sur une grille 7×7 : elle est GROSSIÈRE ;
* elle indique où le modèle "regarde", pas la cause médicale ;
* la description "droite/gauche" suppose une radiographie de face à l'orientation standard
  (côté droit du patient à gauche de l'image). Toujours à vérifier par le professionnel.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .preprocess import center_crop_square

# Pathologies dont la zone attendue est centrale / diffuse : on évite de sur-préciser.
_CENTRAL_TARGETS = {"Cardiomegaly"}


def peak_region(heat: np.ndarray, rel_threshold: float = 0.5) -> dict[str, float]:
    """Barycentre (x, y ∈ [0,1]) et étendue de la zone dépassant ``rel_threshold`` × maximum."""
    h, w = heat.shape
    peak = float(heat.max())
    if peak <= 1e-8:
        return {"cx": 0.5, "cy": 0.5, "area": 0.0}
    mask = heat >= rel_threshold * peak
    ys, xs = np.nonzero(mask)
    return {
        "cx": float(xs.mean() / (w - 1)),
        "cy": float(ys.mean() / (h - 1)),
        "area": float(mask.mean()),
    }


def describe_zone(cx: float, cy: float, target: str = "") -> str:
    """Zone en français à partir du barycentre (convention radiologique de face)."""
    if target in _CENTRAL_TARGETS:
        return "la région médiane du thorax (silhouette cardio-médiastinale)"
    vertical = "supérieur" if cy < 0.33 else ("moyen" if cy < 0.66 else "inférieur")
    if 0.40 <= cx <= 0.60:
        side = "la région médiane"
        return f"{side}, à hauteur du tiers {vertical}"
    side = "droit" if cx < 0.40 else "gauche"  # image gauche = côté droit du patient
    return f"le tiers {vertical} de l'hémithorax {side}"


def score_band(p: float) -> str:
    if p >= 0.60:
        return "élevé"
    if p >= 0.30:
        return "intermédiaire"
    return "faible"


def build_sentence(label: str, target: str, p: float, calibrated: bool, heat: np.ndarray | None) -> str:
    """Phrase explicative prudente — jamais de cause médicale affirmée."""
    band = score_band(p)
    score_txt = f"{p:.0%}" if calibrated else f"{p:.2f} (score brut, non calibré)"
    if heat is None or band == "faible":
        return f"Score {band} pour {label} ({score_txt}) : aucune zone d'attention marquée à signaler."
    region = peak_region(heat)
    zone = describe_zone(region["cx"], region["cy"], target)
    return (
        f"Score {band} pour {label} ({score_txt}), principalement associé à une zone d'attention "
        f"située dans {zone}. Ceci indique où le modèle s'est concentré, pas une cause médicale : "
        f"à vérifier par le professionnel de santé."
    )


def overlay_heatmap(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.45, size: int = 448) -> Image.Image:
    """Superpose la carte de chaleur sur la zone analysée (carré central de l'image)."""
    square = center_crop_square(gray)
    base = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    hm = cv2.resize(heat.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, 1.0)
    colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    weight = (alpha * hm)[..., None]  # transparent là où l'activation est nulle
    blended = (1.0 - weight) * base_rgb + weight * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
