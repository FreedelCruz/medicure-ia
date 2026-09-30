"""Images SYNTHÉTIQUES pour tester la logique (pas des radiographies réelles).

``fake_chest`` dessine un thorax schématique : fond, corps, deux champs pulmonaires sombres,
médiastin clair, côtes (stries) et bruit de texture. Suffisant pour vérifier que le contrôle
qualité, le refus et la priorité se comportent comme prévu.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medguard.synthetic import fake_chest  # noqa: E402


@pytest.fixture
def chest_gray():
    return fake_chest()


@pytest.fixture
def chest_img(chest_gray):
    return Image.fromarray(chest_gray)


@pytest.fixture
def blurry_img(chest_gray):
    return Image.fromarray(cv2.GaussianBlur(chest_gray, (0, 0), 14))


@pytest.fixture
def dark_img(chest_gray):
    return Image.fromarray((chest_gray.astype(np.float32) * 0.25).astype(np.uint8))


@pytest.fixture
def lowres_img(chest_gray):
    return Image.fromarray(cv2.resize(chest_gray, (96, 96), interpolation=cv2.INTER_AREA))


@pytest.fixture
def color_photo_img():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    return Image.fromarray(cv2.GaussianBlur(base, (0, 0), 6))
