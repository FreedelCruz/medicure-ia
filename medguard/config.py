"""Configuration centrale du MedGuard Engine.

Toutes les valeurs numériques ci-dessous sont des VALEURS DE DÉPART pour le MVP.
Elles doivent être recalibrées sur les données de validation (notebooks 01, 03, 04)
puis écrasées via ``artifacts/thresholds.json`` (voir ``load_settings``).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Pathologies ciblées (noms exacts utilisés par TorchXRayVision) -> libellé français.
TARGETS: dict[str, str] = {
    "Effusion": "épanchement pleural",
    "Cardiomegaly": "cardiomégalie",
    "Lung Opacity": "opacité pulmonaire",
}

# Poids pré-entraînés TorchXRayVision.
# ATTENTION fuite de données : les poids "-all" ont été entraînés sur plusieurs datasets
# publics (dont NIH et CheXpert). Évaluer sur ces mêmes datasets donnerait des résultats
# optimistes. Voir docs/KAGGLE.md pour la stratégie d'évaluation externe.
MODEL_WEIGHTS = "densenet121-res224-chex"
INPUT_SIZE = 224

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
# Surchargeable (ex. MEDGUARD_AUDIT_LOG=/data/audit_log.jsonl avec le stockage persistant HF Spaces).
AUDIT_LOG_PATH = Path(os.environ.get("MEDGUARD_AUDIT_LOG", str(ROOT_DIR / "artifacts" / "audit_log.jsonl")))

# Garde-fou pour un déploiement public : refuse une image absurdement grande avant tout traitement.
MAX_INPUT_PIXELS = int(os.environ.get("MEDGUARD_MAX_PIXELS", 30_000_000))  # ~ 6000x5000


@dataclass
class QualityThresholds:
    """Bornes (mauvais, bon) de chaque métrique de qualité. Score 0 au 'mauvais', 1 au 'bon'."""

    # Variance du Laplacien mesurée sur l'image redimensionnée à 512 px.
    blur: tuple[float, float] = (10.0, 60.0)
    # Étendue dynamique p99 - p1 (niveaux de gris 0-255).
    contrast: tuple[float, float] = (50.0, 130.0)
    # Écart absolu de la luminosité moyenne à 128.
    brightness_dev: tuple[float, float] = (95.0, 50.0)
    # Plus petit côté de l'image, en pixels.
    resolution: tuple[float, float] = (128.0, 512.0)
    # Part de pixels >= 250 (surexposition).
    overexposed_fraction: tuple[float, float] = (0.35, 0.10)


@dataclass
class ReliabilityWeights:
    """Poids des trois composantes de l'indice de fiabilité (somme normalisée)."""

    quality: float = 0.35
    decisiveness: float = 0.40  # netteté de la décision calibrée (loin de 50/50)
    stability: float = 0.25  # stabilité du score sous perturbations mineures


@dataclass
class Settings:
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    weights: ReliabilityWeights = field(default_factory=ReliabilityWeights)

    # Règle de fiabilité (section 5.4 du cahier des charges).
    reliability_high: float = 80.0  # >= : confiance haute
    reliability_refuse: float = 50.0  # <  : refus de conclure

    # Axe "suspicion clinique" : seuils sur la probabilité calibrée maximale.
    suspicion_high: float = 0.60
    suspicion_medium: float = 0.30

    # Priorité plancher d'un cas refusé ("Intermédiaire" par défaut : un cas incertain n'est
    # jamais neutre). Mettre "Élevée" pour reproduire à la lettre la règle initiale du
    # cahier des charges (incertain => priorité élevée).
    refused_min_priority: str = "Intermédiaire"

    # Détecteur de type d'examen.
    exam_min_confidence: float = 0.55

    # Perturbations (TTA) : écart-type au-delà duquel la stabilité tombe à 0.
    stability_max_std: float = 0.15


def load_settings(path: str | Path | None = None) -> Settings:
    """Charge les réglages par défaut, surchargés par ``artifacts/thresholds.json`` s'il existe."""
    settings = Settings()
    path = Path(path) if path else ARTIFACTS_DIR / "thresholds.json"
    if not path.exists():
        return settings
    data = json.loads(path.read_text(encoding="utf-8"))
    for f in fields(settings):
        if f.name in ("quality", "weights") or f.name not in data:
            continue
        setattr(settings, f.name, data[f.name])
    for group_name, cls in (("quality", QualityThresholds), ("weights", ReliabilityWeights)):
        for key, value in data.get(group_name, {}).items():
            if hasattr(getattr(settings, group_name), key):
                if group_name == "quality":
                    value = tuple(value)
                setattr(getattr(settings, group_name), key, value)
    return settings


def save_settings(settings: Settings, path: str | Path | None = None) -> Path:
    path = Path(path) if path else ARTIFACTS_DIR / "thresholds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
