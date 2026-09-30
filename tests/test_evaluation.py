import numpy as np

from medguard.evaluation import (
    aurc,
    auroc,
    bootstrap_ci,
    risk_coverage_curve,
    selective_metrics,
    sensitivity_specificity,
    threshold_for_coverage,
    threshold_for_sensitivity,
    youden_threshold,
)


def test_auroc_perfect_random_and_ties():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_auroc_matches_sklearn_definition_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.random(300) + 0.5 * y
    pos, neg = s[y == 1], s[y == 0]
    manual = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(auroc(y, s) - manual) < 1e-9


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.6 * y
    est, lo, hi = bootstrap_ci(auroc, y, s, n_boot=300)
    assert lo <= est <= hi and 0.5 < lo


def test_sensitivity_specificity_and_thresholds():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05])
    sens, spec = sensitivity_specificity(y, s, 0.5)
    assert sens == 0.75 and spec == 0.75
    assert threshold_for_sensitivity(y, s, 1.0) <= 0.3
    assert 0.3 <= youden_threshold(y, s) <= 0.7  # ex aequo entre 0,3 et 0,7


def test_risk_coverage_improves_when_reliability_is_informative():
    rng = np.random.default_rng(4)
    n = 2000
    reliability = rng.random(n) * 100
    correct = rng.random(n) < (0.6 + 0.4 * reliability / 100)  # plus fiable => plus souvent juste
    curve = risk_coverage_curve(correct, reliability)
    assert curve["risk"][int(0.3 * n)] < curve["risk"][-1]  # refuser 70 % des cas baisse le risque
    m = selective_metrics(correct, reliability, threshold=60)
    assert m["accuracy_kept"] > m["accuracy_all"] and m["coverage"] < 1


def test_aurc_is_lower_for_informative_than_random_reliability():
    rng = np.random.default_rng(5)
    n = 3000
    good_rel = rng.random(n)
    correct = rng.random(n) < (0.5 + 0.5 * good_rel)
    random_rel = rng.random(n)
    assert aurc(correct, good_rel) < aurc(correct, random_rel)


def test_threshold_for_coverage():
    rel = np.arange(100)
    t = threshold_for_coverage(rel, 0.8)
    assert selective_metrics(np.ones(100), rel, t)["coverage"] >= 0.8
