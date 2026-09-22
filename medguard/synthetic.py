"""Images SYNTHÉTIQUES (pas des radiographies) pour tester la logique et les notebooks hors ligne.

``fake_chest`` dessine un thorax schématique : fond, corps, deux champs pulmonaires sombres,
médiastin clair, côtes (stries) et bruit de texture.
"""
import cv2
import numpy as np


def fake_chest(size: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 35, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (int(size * 0.42), int(size * 0.48)), 0, 0, 360, 150, -1)  # corps
    cv2.ellipse(img, (int(size * 0.33), int(size * 0.45)), (int(size * 0.14), int(size * 0.30)), 0, 0, 360, 70, -1)
    cv2.ellipse(img, (int(size * 0.67), int(size * 0.45)), (int(size * 0.14), int(size * 0.30)), 0, 0, 360, 70, -1)
    cv2.ellipse(img, (size // 2, int(size * 0.52)), (int(size * 0.07), int(size * 0.20)), 0, 0, 360, 200, -1)  # médiastin
    for y in range(int(size * 0.2), int(size * 0.7), max(size // 24, 4)):  # côtes
        cv2.line(img, (int(size * 0.2), y), (int(size * 0.8), y + size // 40), 190, 2)
    noisy = img.astype(np.float32) + rng.normal(0, 6, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
