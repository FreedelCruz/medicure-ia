import io
import json

import pytest
from PIL import Image

from app.formatting import explanations_md, quality_md, scores_md, status_md
from medguard.calibration import Calibrator
from medguard.mock import MockPredictor
from medguard.pipeline import MedGuardPipeline

SEVERE = {"Effusion": 0.97, "Cardiomegaly": 0.03, "Lung Opacity": 0.04}
AMBIGUOUS = {"Effusion": 0.5, "Cardiomegaly": 0.5, "Lung Opacity": 0.5}


def _pipe(scores):
    return MedGuardPipeline(MockPredictor(scores=scores), calibrator=Calibrator())


def test_status_shows_reliability_and_priority_first(chest_img):
    md = status_md(_pipe(SEVERE).analyze(chest_img))
    assert "MODE SIMULATION" in md and "Indice de fiabilité" in md and "Priorité" in md and "🔴" in md


def test_refused_case_hides_scores_and_explanations(chest_img):
    r = _pipe(AMBIGUOUS).analyze(chest_img)
    assert "Refus de conclure" in status_md(r)
    assert "Aucun score" in scores_md(r) and explanations_md(r) == ""


def test_unsupported_image_message(color_photo_img):
    md = status_md(_pipe(SEVERE).analyze(color_photo_img))
    assert "non pris en charge" in md


def test_scores_table_and_quality(chest_img):
    r = _pipe(SEVERE).analyze(chest_img)
    assert "épanchement pleural" in scores_md(r) and "non calibrés" in scores_md(r)
    assert "exploitable" in quality_md(r)


def test_gradio_ui_builds(chest_img):
    gradio = pytest.importorskip("gradio")
    from app.gradio_app import build_ui

    assert isinstance(build_ui(_pipe(SEVERE)), gradio.Blocks)


def test_api_analyze_and_decision(chest_img, tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import medguard.audit as audit
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr("app.api.log_analysis", lambda r, user="x": audit.log_analysis(r, user, tmp_path / "audit.jsonl"))
    monkeypatch.setattr("app.api.log_human_decision", lambda *a, **k: audit.log_human_decision(*a, **k, path=tmp_path / "audit.jsonl"))
    from app.api import create_app

    client = TestClient(create_app(_pipe(SEVERE)))
    assert client.get("/health").json()["status"] == "ok"

    buf = io.BytesIO(); chest_img.save(buf, format="PNG")
    r = client.post("/analyze", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json(); json.dumps(body)
    assert body["status"] == "ok" and body["decision"]["priority"] == "Élevée"

    ok = client.post("/decision", json={"analysis_id": body["analysis_id"], "decision": "validé"})
    assert ok.status_code == 200
    bad = client.post("/decision", json={"analysis_id": body["analysis_id"], "decision": "?"})
    assert bad.status_code == 422
    assert client.post("/analyze", files={"file": ("x.png", b"pas une image", "image/png")}).status_code == 422


def test_gradio_errors_are_shown_as_readable_message_not_silent(chest_img, capsys):
    pytest.importorskip("gradio")
    from app.gradio_app import safe_analyze

    class BrokenPredictor(MockPredictor):
        def predict(self, grays):
            raise RuntimeError("Numpy is not available")

    pipe = MedGuardPipeline(BrokenPredictor(), calibrator=Calibrator())
    out = safe_analyze(pipe, chest_img, "dr_test")
    assert len(out) == 7
    assert "Erreur technique" in out[0] and "Numpy is not available" in out[0]
    assert "RuntimeError" in capsys.readouterr().err  # traceback aussi dans le terminal


def test_safe_analyze_without_image_asks_for_upload():
    pytest.importorskip("gradio")
    from app.gradio_app import safe_analyze

    assert "Chargez" in safe_analyze(_pipe(SEVERE), None, "x")[0]
