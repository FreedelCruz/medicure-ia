"""Audit trail : qui a fait quoi, quand — sans jamais stocker l'image ni d'identifiant patient.

Chaque ligne du fichier JSONL est un événement indépendant :
* ``analysis``        : une analyse a été produite (empreinte de l'image, statut, fiabilité, priorité) ;
* ``human_decision``  : le professionnel a validé / rejeté / corrigé (journalisation SEULE).

Aucun réentraînement automatique n'est déclenché par ces événements (risque 12.4 du cahier des
charges : une correction humaine erronée ne doit pas être absorbée comme vérité).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AUDIT_LOG_PATH

VALID_DECISIONS = ("validé", "rejeté", "corrigé", "second avis demandé")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(event: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else AUDIT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def log_analysis(result, user: str = "anonyme", path: Path | None = None) -> dict[str, Any]:
    """Journalise une analyse. ``result`` est un ``pipeline.AnalysisResult``."""
    event = {
        "event": "analysis",
        "ts": _now(),
        "user": user,
        "analysis_id": result.analysis_id,
        "image_sha256": result.image_sha256,
        "status": result.status,
        "reliability_index": result.reliability.index if result.reliability else None,
        "priority": result.decision.priority if result.decision else None,
        "model": result.model_info.get("weights"),
        "simulation": bool(result.model_info.get("simulation", False)),
    }
    return _append(event, path)


def log_human_decision(
    analysis_id: str,
    decision: str,
    user: str = "anonyme",
    comment: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Décision invalide : {decision!r} (attendu : {VALID_DECISIONS})")
    event = {
        "event": "human_decision",
        "ts": _now(),
        "user": user,
        "analysis_id": analysis_id,
        "decision": decision,
        "comment": comment[:500],
    }
    return _append(event, path)


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else AUDIT_LOG_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
