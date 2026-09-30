import cv2
import numpy as np
import pytest

from medguard.preprocess import apply_perturbation, center_crop_square, default_perturbations, preprocess_gray
from medguard.splits import group_split, load_splits, save_splits


def test_group_split_never_splits_a_patient_and_is_reproducible():
    rng = np.random.default_rng(0)
    patients = rng.integers(0, 300, 2000)  # ~6-7 images par patient
    a = group_split(patients, seed=7)
    b = group_split(patients, seed=7)
    assert (a == b).all()
    for p in np.unique(patients):
        assert len(set(a[patients == p])) == 1
    props = {n: float((a == n).mean()) for n in ("dev", "calibration", "test")}
    assert abs(props["test"] - 0.5) < 0.08 and abs(props["dev"] - 0.2) < 0.08


def test_group_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        group_split([1, 2, 3], {"a": 0.5, "b": 0.2})


def test_splits_round_trip(tmp_path):
    ids = np.array(["a", "b", "c", "d"])
    names = np.array(["test", "test", "dev", "calibration"])
    path = save_splits(tmp_path / "s.json", ids, names, meta={"seed": 1})
    loaded = load_splits(path)
    assert loaded["test"] == ["a", "b"] and loaded["dev"] == ["c"]


def test_preprocess_shape_dtype_range(chest_gray):
    x = preprocess_gray(chest_gray)
    assert x.shape == (1, 224, 224) and x.dtype == np.float32
    assert -1024.0 <= x.min() and x.max() <= 1024.0


def test_center_crop_is_square_for_rectangular_input():
    assert center_crop_square(np.zeros((300, 500), np.uint8)).shape == (300, 300)


def test_perturbations_keep_shape_and_stay_mild(chest_gray):
    variants = default_perturbations(chest_gray)
    assert len(variants) == 5
    for v in variants:
        assert v.shape == chest_gray.shape and v.dtype == np.uint8
        # Comparaison en basse résolution : le bruit pixel à pixel de l'image synthétique n'a pas de sens ici.
        a = cv2.resize(v, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
        b = cv2.resize(chest_gray, (32, 32), interpolation=cv2.INTER_AREA).astype(float)
        assert np.abs(a - b).mean() < 20  # borne large : l’image synthétique a des contours très nets


@pytest.mark.parametrize("kind,strength", [("jpeg", 60), ("noise", 5), ("crop", 0.05), ("blur", 2.0), ("brightness", 0.9)])
def test_named_perturbations(chest_gray, kind, strength):
    assert apply_perturbation(chest_gray, kind, strength).shape == chest_gray.shape
