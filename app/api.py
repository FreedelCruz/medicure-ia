"""API FastAPI (optionnelle) : mêmes traitements que l'interface, pour intégration.

    uvicorn app.api:app --reload          # vrai modèle
    MEDGUARD_MOCK=1 uvicorn app.api:app   # simulation
"""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from medguard.audit import VALID_DECISIONS, log_analysis, log_human_decision
from medguard.pipeline import MedGuardPipeline, build_default_pipeline

MAX_BYTES = 15 * 1024 * 1024


class DecisionIn(BaseModel):
    analysis_id: str
    decision: str
    user: str = "anonyme"
    comment: str = ""


def create_app(pipeline: MedGuardPipeline | None = None) -> FastAPI:
    api = FastAPI(title="Médicure AI — MedGuard API", version="0.1.0")
    state = {"pipeline": pipeline}

    def get_pipeline() -> MedGuardPipeline:
        if state["pipeline"] is None:  # chargement paresseux (le modèle est lourd)
            state["pipeline"] = build_default_pipeline(mock=os.environ.get("MEDGUARD_MOCK") == "1")
        return state["pipeline"]

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["pipeline"] is not None}

    @api.post("/analyze")
    async def analyze(file: UploadFile = File(...), user: str = "anonyme"):
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Fichier trop volumineux (15 Mo max).")
        try:
            result = get_pipeline().analyze(data)
        except Exception as exc:  # image illisible, format non pris en charge...
            raise HTTPException(422, f"Image illisible ou non prise en charge : {exc}") from exc
        log_analysis(result, user=user)
        return result.to_dict()

    @api.post("/decision")
    def decision(body: DecisionIn):
        if body.decision not in VALID_DECISIONS:
            raise HTTPException(422, f"Décision invalide. Valeurs : {list(VALID_DECISIONS)}")
        return log_human_decision(body.analysis_id, body.decision, body.user, body.comment)

    return api


app = create_app()
