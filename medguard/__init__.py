"""MedGuard Engine — le cœur de confiance de Médicure AI.

    « L'IA observe. Le professionnel de santé décide. »
"""
from .calibration import Calibrator, expected_calibration_error
from .config import TARGETS, Settings, load_settings
from .pipeline import AnalysisResult, MedGuardPipeline, build_default_pipeline
from .quality import QualityReport, assess_quality
from .reliability import Decision, ReliabilityReport, compute_reliability, decide

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "Calibrator",
    "Decision",
    "MedGuardPipeline",
    "QualityReport",
    "ReliabilityReport",
    "Settings",
    "TARGETS",
    "assess_quality",
    "build_default_pipeline",
    "compute_reliability",
    "decide",
    "expected_calibration_error",
    "load_settings",
]
