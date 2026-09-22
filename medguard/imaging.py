"""Chargement et conversions d'images (aucune dépendance à torch)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageLike = Union[str, Path, bytes, np.ndarray, Image.Image]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(src: ImageLike) -> Image.Image:
    """Charge une image depuis un chemin, des octets, un tableau numpy ou une image PIL.

    Le MVP ne gère que JPG/JPEG/PNG (DICOM prévu en phase 2, avec traitement dédié de
    ses métadonnées).
    """
    if isinstance(src, Image.Image):
        return src.copy()
    if isinstance(src, (str, Path)):
        path = Path(src)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Format non pris en charge : {path.suffix} (JPG, JPEG, PNG uniquement).")
        return Image.open(path).copy()
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src))).copy()
    if isinstance(src, np.ndarray):
        arr = src
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Type d'entrée non supporté : {type(src)!r}")


def to_gray(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en tableau uint8 (H, W) niveaux de gris."""
    return np.asarray(img.convert("L"), dtype=np.uint8)


def to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def image_sha256(img: Image.Image) -> str:
    """Empreinte stable de l'image (pixels), utilisée pour l'audit sans stocker l'image."""
    arr = to_gray(img)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()
