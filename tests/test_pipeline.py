import json

import numpy as np
import pytest

from medguard.audit import log_analysis, log_human_decision, read_log
from medguard.calibration import Calibrator
from medguard.mock import MockPredictor
from medguard.pipeline import MedGuardPipeline, build_default_pipeline
from medguard.report import render_html

CONFIDENT_SEVERE = {"Effusion": 0.97, "Cardiomegaly": 0.03, "Lung Opacity": 0.04}
AMBIGUOUS = {"Effusion": 0.52, "Cardiomegaly": 0.47, "Lung Opacity": 0.50}


class ExplodingPredictor(MockPredictor):
    """Échoue si on l'appelle : prouve que le modèle n'est PAS consulté sur une image inexploitable."""

    def predict(self, grays):
        raise AssertionError("Le modèle ne doit pas être appelé.")


def _pipe(scores=None, jitter=0.0, calibrator=None):
    return MedGuardPipeline(MockPredictor(scores=scores, jitter=jitter), calibrator=calibrator or Calibrator())


def test_confident_severe_case_full_flow(chest_img):
    r = _pipe(CONFIDENT_SEVERE).analyze(chest_img)
    assert r.status == "ok" and r.scores_visible
    assert r.decision.priority == "Élevée"
    assert "Effusion" in r.overlays and "Effusion" in r.heatmaps
    assert "Cardiomegaly" not in r.overlays  # score bas : pas de carte
    assert "principalement associé" in r.explanations["Effusion"]
    assert "pas une cause médicale" in r.explanations["Effusion"]
    assert any("NON calibrés" in m for m in r.messages)  # calibrateur vide => avertissement


def test_calibrated_scores_display_as_percent(chest_img):
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    cal = Calibrator(fitted_on="jeu de test synthétique").fit(
        {k: p for k in CONFIDENT_SEVERE}, {k: y for k in CONFIDENT_SEVERE}, "platt"
    )
    r = _pipe(CONFIDENT_SEVERE, calibrator=cal).analyze(chest_img)
    assert all(t.is_calibrated for t in r.targets)
    assert r.targets[0].display.endswith("%")
    assert r.calibration_info == "jeu de test synthétique"


def test_blurry_image_is_refused_without_calling_the_model(blurry_img):
    r = MedGuardPipeline(ExplodingPredictor(), calibrator=Calibrator()).analyze(blurry_img)
    assert r.status == "refused"
    assert not r.scores_visible and r.to_dict()["targets"] == []
    assert r.decision.human_review_required and r.decision.priority != "Faible"
    assert any("floue" in m for m in r.messages)


def test_non_chest_image_is_unsupported(color_photo_img):
    r = MedGuardPipeline(ExplodingPredictor(), calibrator=Calibrator()).analyze(color_photo_img)
    assert r.status == "unsupported" and r.decision is None
    assert any("non pris en charge" in m for m in r.messages)


def test_ambiguous_case_is_refused_and_scores_are_hidden(chest_img):
    r = _pipe(AMBIGUOUS).analyze(chest_img)
    assert r.status == "refused" and not r.scores_visible
    assert r.decision.priority in ("Intermédiaire", "Élevée")
    assert r.explanations == {} and r.overlays == {}
    assert "vérification humaine" in r.messages[0]


def test_unstable_model_lowers_reliability(chest_img):
    stable = _pipe(CONFIDENT_SEVERE, jitter=0.0).analyze(chest_img)
    shaky = _pipe(CONFIDENT_SEVERE, jitter=0.2).analyze(chest_img)
    assert shaky.reliability.components["stability"] < stable.reliability.components["stability"]
    assert shaky.reliability.index < stable.reliability.index


def test_result_is_json_serialisable(chest_img):
    d = _pipe(CONFIDENT_SEVERE).analyze(chest_img).to_dict()
    json.dumps(d, ensure_ascii=False)  # ne doit pas lever


def test_audit_trail_logs_analysis_and_human_decision(tmp_path, chest_img):
    log = tmp_path / "audit.jsonl"
    r = _pipe(CONFIDENT_SEVERE).analyze(chest_img)
    log_analysis(r, user="dr_test", path=log)
    log_human_decision(r.analysis_id, "rejeté", user="dr_test", comment="désaccord", path=log)
    events = read_log(log)
    assert [e["event"] for e in events] == ["analysis", "human_decision"]
    assert events[0]["image_sha256"] == r.image_sha256 and events[0]["simulation"] is True
    assert events[1]["decision"] == "rejeté"
    assert "image" not in json.dumps(events).replace("image_sha256", "")  # aucune image stockée
    with pytest.raises(ValueError):
        log_human_decision(r.analysis_id, "peut-être", path=log)


def test_report_puts_reliability_before_scores_and_has_no_precommitted_decision(chest_img):
    r = _pipe(CONFIDENT_SEVERE).analyze(chest_img)
    html = render_html(r)
    assert html.index("indice de fiabilité") < html.index("Résultats par anomalie")
    assert "MODE SIMULATION" in html
    assert "Aucune décision enregistrée" in html
    assert "NON diagnostique" in html
    with_decision = render_html(r, {"decision": "validé", "user": "dr_test", "comment": ""})
    assert "validé" in with_decision


def test_refused_report_shows_no_scores(chest_img):
    html = render_html(_pipe(AMBIGUOUS).analyze(chest_img))
    assert "Aucun score n'est présenté" in html
    assert "Résultats par anomalie" not in html


def test_oversized_image_is_rejected_before_any_processing():
    import numpy as np
    from PIL import Image

    huge = Image.fromarray(np.zeros((7000, 7000), dtype=np.uint8))  # 49 Mpx > 30 Mpx par défaut
    with pytest.raises(ValueError, match="trop grande"):
        MedGuardPipeline(ExplodingPredictor(), calibrator=Calibrator()).analyze(huge)


def test_build_default_pipeline_mock_works(chest_img):
    r = build_default_pipeline(mock=True, calibrator=Calibrator()).analyze(chest_img)
    assert r.status in ("ok", "refused")
