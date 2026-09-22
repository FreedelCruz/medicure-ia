"""Mise en forme (Markdown) des résultats pour l'interface. Sans dépendance à Gradio."""
from __future__ import annotations

from medguard.pipeline import AnalysisResult
from medguard.reliability import PRIORITY_EMOJI

DISCLAIMER = (
    "**Outil d'aide à la décision, non diagnostique — prototype.** "
    "L'IA observe ; le professionnel de santé décide."
)


def simulation_banner(result: AnalysisResult) -> str:
    if result.model_info.get("simulation"):
        return "### 🧪 MODE SIMULATION — scores fictifs, aucune valeur médicale\n\n"
    return ""


def status_md(result: AnalysisResult) -> str:
    """Statut, fiabilité et priorité — affichés AVANT les scores (limite l'ancrage sur un chiffre)."""
    out = [simulation_banner(result)]
    if result.status == "unsupported":
        out.append("## ❌ Type d'image non pris en charge\n")
        out += [f"- {m}" for m in result.messages[1:]] or ["- Une radiographie thoracique est attendue."]
        return "\n".join(out)

    r, d = result.reliability, result.decision
    if result.status == "refused":
        out.append("## ⛔ Refus de conclure — vérification humaine nécessaire\n")
    else:
        out.append("## ✅ Analyse disponible\n")
    if r:
        out.append(f"**Indice de fiabilité : {r.index:.0f}/100** ({r.level})  ")
        comps = " · ".join(f"{k} {v:.0f}" for k, v in r.components.items())
        out.append(f"<sub>Composantes : {comps}</sub>\n")
    if d:
        out.append(f"**Priorité : {PRIORITY_EMOJI[d.priority]} {d.priority}**")
        if d.second_opinion_required:
            out.append(" — second avis requis")
        out.append("\n")
        out += [f"- {m}" for m in d.reasons]
    if result.status == "refused" and result.quality and result.quality.fatal_issues:
        out += [f"- {m}" for m in result.quality.fatal_issues if m not in (d.reasons if d else [])]
    return "\n".join(out)


def quality_md(result: AnalysisResult) -> str:
    if result.quality is None:
        return ""
    q = result.quality
    lines = [f"**Qualité de l'image : {q.score:.0f}/100** — " + ("exploitable" if q.usable else "non exploitable")]
    lines += [f"- {m}" for m in q.issues]
    return "\n".join(lines)


def scores_md(result: AnalysisResult) -> str:
    if not result.scores_visible:
        return "_Aucun score n'est présenté : le système ne conclut pas sur ce cas._"
    lines = ["| Anomalie ciblée | Score | Stabilité (écart-type) |", "|---|---|---|"]
    for t in result.targets:
        lines.append(f"| {t.label} | {t.display} | {t.std:.3f} |")
    if not all(t.is_calibrated for t in result.targets):
        lines.append("\n⚠️ Les scores marqués « non calibrés » ne sont pas des probabilités.")
    return "\n".join(lines)


def explanations_md(result: AnalysisResult) -> str:
    if not result.scores_visible:
        return ""
    lines = [f"- {result.explanations[t.name]}" for t in result.targets if t.name in result.explanations]
    lines.append(
        "\n<sub>Les cartes de chaleur sont grossières et montrent où le modèle a regardé, pas la cause médicale. "
        "Orientation supposée standard.</sub>"
    )
    return "\n".join(lines)
