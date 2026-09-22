"""Orchestration du MedGuard Engine.

    IMAGE → type d'examen → qualité → prédiction (+ perturbations) → calibration
          → indice de fiabilité → décision (statut + priorité) → explication

L'ordre est volontaire : on ne laisse parler le modèle de diagnostic que si l'image est
d'un type pris en charge ET exploitable.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .calibration import Calibrator
from .config import ARTIFACTS_DIR, MAX_INPUT_PIXELS, TARGETS, Settings, load_settings
from .exam_check import ExamCheck, check_exam_type
from .explain import build_sentence, overlay_heatmap
from .imaging import ImageLike, image_sha256, load_image, to_gray
from .model import Predictor
from .preprocess import default_perturbations
from .quality import QualityReport, assess_quality
from .reliability import Decision, ReliabilityReport, compute_reliability, decide


@dataclass
class TargetResult:
    name: str  # nom TorchXRayVision
    label: str  # libellé français
    raw_score: float
    calibrated: float
    is_calibrated: bool
    std: float  # écart-type du score calibré sous perturbations
    display: str  # texte à afficher ("72 %" ou "score brut 0,61 (non calibré)")


@dataclass
class AnalysisResult:
    analysis_id: str
    created_at: str
    image_sha256: str
    status: str  # "unsupported" | "refused" | "ok"
    exam: ExamCheck
    quality: Optional[QualityReport]
    targets: list[TargetResult] = field(default_factory=list)
    reliability: Optional[ReliabilityReport] = None
    decision: Optional[Decision] = None
    explanations: dict[str, str] = field(default_factory=dict)
    heatmaps: dict[str, np.ndarray] = field(default_factory=dict)
    overlays: dict[str, Image.Image] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    model_info: dict = field(default_factory=dict)
    calibration_info: str = ""

    @property
    def scores_visible(self) -> bool:
        """Les scores ne sont montrés que si le système accepte de conclure."""
        return self.status == "ok"

    def to_dict(self) -> dict:
        """Résumé JSON-sérialisable (sans images ni cartes)."""
        return {
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "image_sha256": self.image_sha256,
            "status": self.status,
            "messages": self.messages,
            "exam": vars(self.exam),
            "quality": vars(self.quality) if self.quality else None,
            "reliability": vars(self.reliability) if self.reliability else None,
            "decision": vars(self.decision) if self.decision else None,
            "targets": [vars(t) for t in self.targets] if self.scores_visible else [],
            "explanations": self.explanations if self.scores_visible else {},
            "elapsed_s": round(self.elapsed_s, 2),
            "model_info": self.model_info,
            "calibration_info": self.calibration_info,
        }


class MedGuardPipeline:
    def __init__(
        self,
        predictor: Predictor,
        calibrator: Calibrator | None = None,
        settings: Settings | None = None,
        use_perturbations: bool = True,
    ):
        self.predictor = predictor
        self.calibrator = calibrator if calibrator is not None else Calibrator.load(ARTIFACTS_DIR / "calibration.json")
        self.settings = settings or load_settings()
        self.use_perturbations = use_perturbations

    # ------------------------------------------------------------------ API
    def analyze(self, image: ImageLike, with_explanations: bool = True) -> AnalysisResult:
        """``with_explanations=False`` saute Grad-CAM (évaluations sur des milliers d'images)."""
        t0 = time.perf_counter()
        img = load_image(image)
        if img.width * img.height > MAX_INPUT_PIXELS:
            raise ValueError(
                f"Image trop grande ({img.width}x{img.height} px, "
                f"{MAX_INPUT_PIXELS // 1_000_000} Mpx maximum). Réduisez-la avant de la charger."
            )
        base = dict(
            analysis_id=uuid.uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            image_sha256=image_sha256(img),
            model_info=dict(getattr(self.predictor, "info", {})),
            calibration_info=self.calibrator.fitted_on or "non calibré",
        )

        # 1) Type d'examen
        exam = check_exam_type(img, self.settings.exam_min_confidence)
        if not exam.is_chest_xray:
            return AnalysisResult(
                status="unsupported",
                exam=exam,
                quality=None,
                messages=["❌ Type d'image non pris en charge (radiographie thoracique attendue)."] + exam.reasons,
                elapsed_s=time.perf_counter() - t0,
                **base,
            )

        # 2) Qualité
        quality = assess_quality(img, self.settings.quality)
        if not quality.usable:
            reliability = compute_reliability(quality, {}, {}, self.settings)
            decision = decide(quality, None, reliability, self.settings)
            return AnalysisResult(
                status="refused",
                exam=exam,
                quality=quality,
                reliability=reliability,
                decision=decision,
                messages=["⚠️ Image de qualité insuffisante : analyse non réalisée."] + quality.fatal_issues,
                elapsed_s=time.perf_counter() - t0,
                **base,
            )

        # 3) Prédiction sur l'image + perturbations mineures (mesure de stabilité)
        gray = to_gray(img)
        variants = [gray] + (default_perturbations(gray) if self.use_perturbations else [])
        raw = np.asarray(self.predictor.predict(variants), dtype=float)  # (n, T)
        names = list(self.predictor.targets)

        # 4) Calibration puis écart-type sur les scores calibrés
        cal = np.stack([np.asarray(self.calibrator.calibrate(n, raw[:, j])) for j, n in enumerate(names)], axis=1)
        calibrated = {n: float(cal[0, j]) for j, n in enumerate(names)}
        stds = {n: float(cal[:, j].std()) for j, n in enumerate(names)}

        targets = []
        for j, n in enumerate(names):
            fitted = self.calibrator.is_fitted(n)
            p = calibrated[n]
            targets.append(
                TargetResult(
                    name=n,
                    label=TARGETS.get(n, n),
                    raw_score=float(raw[0, j]),
                    calibrated=p,
                    is_calibrated=fitted,
                    std=stds[n],
                    display=f"{p:.0%}" if fitted else f"score brut {p:.2f} (non calibré)",
                )
            )

        # 5) Fiabilité + décision
        reliability = compute_reliability(quality, calibrated, stds, self.settings)
        decision = decide(quality, calibrated, reliability, self.settings)
        status = "ok" if decision.conclusion_allowed else "refused"

        messages = ["✅ Image exploitable."] + quality.issues
        if not all(t.is_calibrated for t in targets):
            messages.append("⚠️ Scores NON calibrés : ne pas les lire comme des probabilités.")
        if status == "refused":
            messages = [decision.message] + decision.reasons

        result = AnalysisResult(
            status=status,
            exam=exam,
            quality=quality,
            targets=targets,
            reliability=reliability,
            decision=decision,
            messages=messages,
            **base,
        )

        # 6) Explications (seulement si le système accepte de conclure)
        if status == "ok" and with_explanations:
            for t in targets:
                heat = None
                if t.calibrated >= self.settings.suspicion_medium:
                    heat = self.predictor.explain(gray, t.name)
                if heat is not None:
                    result.heatmaps[t.name] = heat
                    result.overlays[t.name] = overlay_heatmap(gray, heat)
                result.explanations[t.name] = build_sentence(t.label, t.name, t.calibrated, t.is_calibrated, heat)

        result.elapsed_s = time.perf_counter() - t0
        return result


def build_default_pipeline(weights: Optional[str] = None, mock: bool = False, **kwargs) -> MedGuardPipeline:
    """Construit le pipeline avec le vrai modèle (torch requis) ou le prédicteur simulé."""
    if mock:
        from .mock import MockPredictor

        return MedGuardPipeline(MockPredictor(), **kwargs)
    from .model import XRVPredictor

    predictor = XRVPredictor(weights) if weights else XRVPredictor()
    return MedGuardPipeline(predictor, **kwargs)
