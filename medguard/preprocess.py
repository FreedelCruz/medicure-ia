"""Prétraitement et perturbations (numpy + OpenCV, aucune dépendance à torch).

Le prétraitement reproduit celui de TorchXRayVision : recadrage carré centré sur la plus
petite dimension, redimensionnement à 224 px, puis mise à l'échelle des niveaux de gris
vers [-1024, 1024].
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import INPUT_SIZE


def center_crop_square(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    s = min(h, w)
    y0, x0 = h // 2 - s // 2, w // 2 - s // 2
    return gray[y0 : y0 + s, x0 : x0 + s]


def preprocess_gray(gray: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """uint8 (H, W) -> float32 (1, size, size) dans [-1024, 1024]."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    sq = center_crop_square(gray)
    small = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    scaled = (2.0 * (small.astype(np.float32) / 255.0) - 1.0) * 1024.0
    return scaled[None, :, :].astype(np.float32)


# ------------------------------------------------------------------ perturbations


def jpeg_compress(gray: np.ndarray, quality: int = 85) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Échec de la compression JPEG")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def add_noise(gray: np.ndarray, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = gray.astype(np.float32) + rng.normal(0.0, sigma, gray.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def crop_rescale(gray: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """Léger recadrage (retire ``margin`` de chaque bord) puis retour à la taille d'origine."""
    h, w = gray.shape
    dy, dx = int(h * margin), int(w * margin)
    cropped = gray[dy : h - dy, dx : w - dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def scale_brightness(gray: np.ndarray, factor: float = 1.05) -> np.ndarray:
    return np.clip(gray.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def gaussian_blur(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigma)


def default_perturbations(gray: np.ndarray) -> list[np.ndarray]:
    """Perturbations MINEURES, sans rapport avec la qualité clinique de l'image.

    Servent à mesurer la stabilité : si le score bouge beaucoup sous ces variations, on ne
    peut pas lui faire confiance (section 8.4 du cahier des charges).
    """
    return [
        scale_brightness(gray, 1.05),
        scale_brightness(gray, 0.95),
        add_noise(gray, 3.0, seed=0),
        jpeg_compress(gray, 85),
        crop_rescale(gray, 0.03),
    ]


PERTURBATION_REGISTRY = {
    "jpeg": lambda g, s: jpeg_compress(g, int(s)),  # s = qualité (ex. 90, 70, 50)
    "noise": lambda g, s: add_noise(g, float(s)),  # s = écart-type
    "crop": lambda g, s: crop_rescale(g, float(s)),  # s = marge (ex. 0.02, 0.05)
    "blur": lambda g, s: gaussian_blur(g, float(s)),  # s = sigma
    "brightness": lambda g, s: scale_brightness(g, float(s)),  # s = facteur
}


def apply_perturbation(gray: np.ndarray, kind: str, strength: float) -> np.ndarray:
    return PERTURBATION_REGISTRY[kind](gray, strength)
