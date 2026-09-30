import numpy as np

from medguard.calibration import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
    logit,
    sigmoid,
)


def _overconfident_data(n=4000, seed=0, temperature=3.0):
    """Modèle sur-confiant : vrais logits z, mais il annonce sigmoid(z * temperature)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < sigmoid(z)).astype(int)
    return sigmoid(z * temperature), y


def test_ece_is_near_zero_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.02


def test_ece_is_high_for_overconfident_scores():
    p, y = _overconfident_data()
    assert expected_calibration_error(p, y) > 0.08


def test_temperature_scaling_reduces_ece_and_finds_t_above_one():
    p, y = _overconfident_data()
    z = logit(p)
    t = fit_temperature(z, y)
    assert 2.0 < t < 4.5  # vrai T = 3
    p_cal = sigmoid(z / t)
    assert expected_calibration_error(p_cal, y) < expected_calibration_error(p, y)
    assert brier_score(p_cal, y) <= brier_score(p, y)


def test_platt_scaling_corrects_a_shifted_and_overconfident_model():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, 6000)
    y = (rng.random(6000) < sigmoid(z)).astype(int)
    p = sigmoid(2.5 * z + 1.0)  # sur-confiant ET biaisé vers le positif
    a, b = fit_platt(logit(p), y)
    assert 0.2 < a < 0.6
    assert expected_calibration_error(sigmoid(a * logit(p) + b), y) < expected_calibration_error(p, y) / 2


def test_calibrator_round_trip(tmp_path):
    p, y = _overconfident_data(n=1500)
    cal = Calibrator(fitted_on="test").fit({"Effusion": p}, {"Effusion": y}, method="platt")
    path = cal.save(tmp_path / "calibration.json")
    loaded = Calibrator.load(path)
    assert loaded.fitted_on == "test"
    assert loaded.is_fitted("Effusion")
    np.testing.assert_allclose(loaded.calibrate("Effusion", p[:20]), cal.calibrate("Effusion", p[:20]))


def test_unfitted_calibrator_is_identity_and_reports_not_fitted():
    cal = Calibrator()
    assert not cal.is_fitted("Effusion")
    np.testing.assert_allclose(cal.calibrate("Effusion", [0.2, 0.9]), [0.2, 0.9])


def test_missing_calibration_file_gives_empty_calibrator(tmp_path):
    assert not Calibrator.load(tmp_path / "absent.json").per_target
