"""Composant 6 — Uncertainty & Reliability.

Deux axes distincts (voir docs/DECISION_LOGIC.md) :

* SUSPICION clinique : probabilité calibrée maximale parmi les pathologies ciblées.
  C'est elle qui détermine la priorité ("un gros épanchement détecté avec certitude n'est
  pas un cas peu prioritaire").
* FIABILITÉ (0-100) : moyenne géométrique pondérée de la qualité d'image, de la netteté de
  la décision et de la stabilité aux perturbations (le maillon le plus faible domine). C'est elle qui déclenche le refus de conclure.

Règle : fiabilité < seuil de refus => aucune conclusion, relecture humaine requise, et la
priorité ne peut pas descendre sous ``refused_min_priority`` (un cas incertain n'est jamais
neutre).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import Settings
from .quality import QualityReport

PRIORITY_ORDER = ["Faible", "Intermédiaire", "Élevée"]
PRIORITY_EMOJI = {"Faible": "🟢", "Intermédiaire": "🟠", "Élevée": "🔴"}


@dataclass
class ReliabilityReport:
    index: float  # 0-100
    level: str  # "haute" | "modérée" | "insuffisante"
    refuse: bool
    components: dict[str, float] = field(default_factory=dict)  # chacune 0-100


@dataclass
class Decision:
    status: str  # "ok" | "refused"
    conclusion_allowed: bool
    suspicion_level: str
    priority: str
    priority_emoji: str
    vigilance: bool
    human_review_required: bool
    second_opinion_required: bool
    message: str
    reasons: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------ composantes


def decisiveness(probs: Mapping[str, float]) -> float:
    """0 si la décision est un coup de pile ou face (p=0,5), 1 si p vaut 0 ou 1.

    Agrégé sur les pathologies par 0,5*min + 0,5*moyenne : la pathologie la plus ambiguë
    pèse davantage que les autres.
    """
    d = np.array([abs(2.0 * float(p) - 1.0) for p in probs.values()])
    if d.size == 0:
        return 0.0
    return float(0.5 * d.min() + 0.5 * d.mean())


def stability(stds: Mapping[str, float], max_std: float) -> float:
    """1 si les scores ne bougent pas sous perturbations mineures, 0 si l'écart-type >= max_std."""
    if not stds:
        return 1.0
    worst = max(float(s) for s in stds.values())
    return float(np.clip(1.0 - worst / max_std, 0.0, 1.0))


def compute_reliability(
    quality: QualityReport,
    calibrated: Mapping[str, float],
    stds: Mapping[str, float],
    settings: Settings,
) -> ReliabilityReport:
    w = settings.weights
    comps = {
        "quality": quality.score / 100.0,
        "decisiveness": decisiveness(calibrated),
        "stability": stability(stds, settings.stability_max_std),
    }
    # Moyenne GÉOMÉTRIQUE pondérée (maillon faible) : une composante proche de zéro fait
    # s'effondrer l'indice. Avec une moyenne arithmétique, une image nette et stable pouvait
    # masquer un score à 50/50 (indice ~60 au lieu d'un refus).
    total_w = w.quality + w.decisiveness + w.stability
    log_sum = sum(
        weight * np.log(max(comps[name], 1e-3))
        for name, weight in (("quality", w.quality), ("decisiveness", w.decisiveness), ("stability", w.stability))
    )
    index = 100.0 * float(np.exp(log_sum / total_w))

    # Une image jugée inexploitable ne peut jamais produire une fiabilité "acceptable".
    if not quality.usable:
        index = min(index, settings.reliability_refuse - 0.1)

    index = float(np.clip(index, 0.0, 100.0))
    if index >= settings.reliability_high:
        level = "haute"
    elif index >= settings.reliability_refuse:
        level = "modérée"
    else:
        level = "insuffisante"

    return ReliabilityReport(
        index=round(index, 1),
        level=level,
        refuse=index < settings.reliability_refuse,
        components={k: round(100.0 * v, 1) for k, v in comps.items()},
    )


# ------------------------------------------------------------------------ priorité


def suspicion_level(calibrated: Mapping[str, float], settings: Settings) -> str:
    top = max((float(p) for p in calibrated.values()), default=0.0)
    if top >= settings.suspicion_high:
        return "Élevée"
    if top >= settings.suspicion_medium:
        return "Intermédiaire"
    return "Faible"


def _max_priority(a: str, b: str) -> str:
    return a if PRIORITY_ORDER.index(a) >= PRIORITY_ORDER.index(b) else b


def decide(
    quality: QualityReport,
    calibrated: Mapping[str, float] | None,
    reliability: ReliabilityReport,
    settings: Settings,
) -> Decision:
    """Combine suspicion et fiabilité en une décision d'affichage (statut + priorité)."""
    calibrated = calibrated or {}
    susp = suspicion_level(calibrated, settings) if calibrated else "Faible"
    reasons: list[str] = []

    if reliability.refuse:
        if not quality.usable:
            reasons.extend(quality.fatal_issues)
        if reliability.components.get("decisiveness", 100) < 40:
            reasons.append("Les scores sont trop proches de l'indécision pour conclure.")
        if reliability.components.get("stability", 100) < 40:
            reasons.append("Les scores varient trop sous de petites perturbations de l'image.")
        priority = _max_priority(susp, settings.refused_min_priority)
        return Decision(
            status="refused",
            conclusion_allowed=False,
            suspicion_level=susp,
            priority=priority,
            priority_emoji=PRIORITY_EMOJI[priority],
            vigilance=True,
            human_review_required=True,
            second_opinion_required=True,
            message="Le niveau de confiance est insuffisant : vérification humaine nécessaire.",
            reasons=reasons or ["Confiance globale insuffisante."],
        )

    vigilance = reliability.level == "modérée"
    if vigilance:
        reasons.append("Confiance modérée : à interpréter avec vigilance.")
    priority = susp
    return Decision(
        status="ok",
        conclusion_allowed=True,
        suspicion_level=susp,
        priority=priority,
        priority_emoji=PRIORITY_EMOJI[priority],
        vigilance=vigilance,
        human_review_required=False,
        second_opinion_required=priority == "Élevée",
        message="Analyse disponible. La décision finale appartient au professionnel de santé.",
        reasons=reasons,
    )
