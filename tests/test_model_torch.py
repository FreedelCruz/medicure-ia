"""Tests du prédicteur TorchXRayVision et de Grad-CAM.

Ignorés automatiquement si torch / torchxrayvision ne sont pas installés. Ils utilisent une
architecture DenseNet à poids ALÉATOIRES (aucun téléchargement) : on vérifie la plomberie
(formes, plages de valeurs, circulation du gradient), pas la qualité médicale.
"""
import numpy as np
import pytest

try:  # un environnement CUDA cassé lève parfois ValueError/OSError, pas seulement ImportError
    import torch  # noqa: F401
    import torchxrayvision as xrv
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torch / torchxrayvision indisponibles : {exc}", allow_module_level=True)

from medguard.model import Predictor, XRVPredictor  # noqa: E402


@pytest.fixture
def predictor(monkeypatch):
    real = xrv.models.DenseNet

    def fake(weights=None, cache_dir=None, **kwargs):
        m = real(weights=None)
        m.pathologies = list(xrv.datasets.default_pathologies)
        return m.eval()

    monkeypatch.setattr(xrv.models, "DenseNet", fake)
    return XRVPredictor("densenet121-res224-chex", device="cpu")


def test_predictor_respects_the_protocol(predictor):
    assert isinstance(predictor, Predictor)
    assert predictor.targets == ["Effusion", "Cardiomegaly", "Lung Opacity"]


def test_predict_shape_and_range(predictor, chest_gray):
    out = predictor.predict([chest_gray, chest_gray])
    assert out.shape == (2, 3) and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_gradcam_shape_and_range(predictor, chest_gray):
    heat = predictor.explain(chest_gray, "Effusion")
    assert heat.shape == (224, 224) and np.isfinite(heat).all()
    assert heat.min() >= 0.0 and heat.max() <= 1.0 + 1e-6


def test_gradcam_unknown_target_returns_none(predictor, chest_gray):
    assert predictor.explain(chest_gray, "Mass") is None


def test_missing_target_raises_helpful_error(monkeypatch):
    real = xrv.models.DenseNet

    def fake(weights=None, cache_dir=None, **kwargs):
        m = real(weights=None)
        m.pathologies = ["Atelectasis"]  # ne prédit pas nos cibles
        return m

    monkeypatch.setattr(xrv.models, "DenseNet", fake)
    with pytest.raises(ValueError, match="ne prédisent pas"):
        XRVPredictor("densenet121-res224-chex", device="cpu")
