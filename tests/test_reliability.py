from medguard.config import Settings
from medguard.quality import assess_quality
from medguard.reliability import compute_reliability, decide, decisiveness, stability

S = Settings()


def _run(quality_img, probs, stds=None):
    q = assess_quality(quality_img)
    stds = stds or {k: 0.0 for k in probs}
    rel = compute_reliability(q, probs, stds, S)
    return q, rel, decide(q, probs, rel, S)


def test_decisiveness_is_zero_at_coin_flip_and_one_at_certainty():
    assert decisiveness({"a": 0.5}) == 0.0
    assert decisiveness({"a": 1.0}) == 1.0
    assert decisiveness({"a": 0.0}) == 1.0


def test_stability_drops_with_perturbation_spread():
    assert stability({"a": 0.0}, 0.15) == 1.0
    assert stability({"a": 0.15}, 0.15) == 0.0
    assert stability({"a": 0.03, "a2": 0.12}, 0.15) < 0.25  # la pire pathologie domine


def test_confident_severe_finding_is_HIGH_priority_not_low(chest_img):
    """Le défaut logique de la première version : haute confiance ne doit PAS donner priorité faible."""
    probs = {"Effusion": 0.97, "Cardiomegaly": 0.03, "Lung Opacity": 0.04}
    _, rel, d = _run(chest_img, probs)
    assert rel.index >= 80 and rel.level == "haute"
    assert d.status == "ok" and d.conclusion_allowed
    assert d.priority == "Élevée"
    assert d.second_opinion_required


def test_confident_normal_case_is_low_priority(chest_img):
    probs = {"Effusion": 0.03, "Cardiomegaly": 0.02, "Lung Opacity": 0.04}
    _, rel, d = _run(chest_img, probs)
    assert d.priority == "Faible" and d.status == "ok"
    assert not d.vigilance and not d.human_review_required


def test_ambiguous_case_refuses_to_conclude_and_is_never_neutral(chest_img):
    probs = {"Effusion": 0.52, "Cardiomegaly": 0.47, "Lung Opacity": 0.5}
    _, rel, d = _run(chest_img, probs)
    assert rel.refuse and rel.index < S.reliability_refuse
    assert d.status == "refused" and not d.conclusion_allowed
    assert d.human_review_required
    assert d.priority != "Faible"


def test_unstable_scores_trigger_refusal_even_if_confident(chest_img):
    probs = {"Effusion": 0.9, "Cardiomegaly": 0.05, "Lung Opacity": 0.06}
    stds = {"Effusion": 0.4, "Cardiomegaly": 0.0, "Lung Opacity": 0.0}
    _, rel, d = _run(chest_img, probs, stds)
    assert rel.components["stability"] == 0.0
    assert d.status == "refused" or rel.level != "haute"


def test_unusable_image_can_never_reach_acceptable_reliability(blurry_img):
    probs = {"Effusion": 0.99, "Cardiomegaly": 0.01, "Lung Opacity": 0.01}
    q, rel, d = _run(blurry_img, probs)
    assert not q.usable
    assert rel.refuse and rel.index < S.reliability_refuse
    assert d.status == "refused"
    assert any("floue" in r for r in d.reasons)


def test_refused_priority_floor_is_configurable(chest_img):
    probs = {"Effusion": 0.5, "Cardiomegaly": 0.5, "Lung Opacity": 0.5}
    q = assess_quality(chest_img)
    strict = Settings(refused_min_priority="Élevée")
    rel = compute_reliability(q, probs, {k: 0.0 for k in probs}, strict)
    assert decide(q, probs, rel, strict).priority == "Élevée"


def test_moderate_reliability_shows_vigilance(chest_img):
    probs = {"Effusion": 0.15, "Cardiomegaly": 0.12, "Lung Opacity": 0.1}
    _, rel, d = _run(chest_img, probs, {k: 0.04 for k in probs})
    if rel.level == "modérée":
        assert d.vigilance and d.status == "ok"
