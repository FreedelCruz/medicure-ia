"""Rapport structuré et téléchargeable (HTML autonome, images embarquées).

Ordre d'affichage volontaire (section 10.1 du cahier des charges) : l'incertitude et la
fiabilité AVANT les scores, pour limiter l'ancrage sur un chiffre. Aucune décision n'est
pré-cochée : la décision humaine n'apparaît que si elle a été saisie.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Optional

from PIL import Image

from .reliability import PRIORITY_EMOJI

DISCLAIMER = (
    "Médicure AI est un outil d'aide à la décision NON diagnostique, en phase de prototype. "
    "Il ne remplace ni un médecin ni un radiologue. L'IA observe ; le professionnel de santé décide."
)

_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1{font-size:1.6rem;margin-bottom:.2rem} h2{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #e5e7eb;padding-bottom:.2rem}
.muted{color:#6b7280;font-size:.9rem} .box{border:1px solid #e5e7eb;border-radius:8px;padding:.8rem 1rem;margin:.6rem 0}
.warn{background:#fff7ed;border-color:#fdba74} .bad{background:#fef2f2;border-color:#fca5a5} .ok{background:#f0fdf4;border-color:#86efac}
table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #e5e7eb;padding:.4rem;text-align:left;font-size:.95rem}
img{max-width:100%;border-radius:6px} .idx{font-size:2rem;font-weight:700}
"""


def _img_tag(img: Image.Image, width: int = 380) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="carte d\'attention" width="{width}" src="data:image/jpeg;base64,{b64}">'


def render_html(result, human_decision: Optional[dict] = None) -> str:
    """``result`` : ``pipeline.AnalysisResult``. ``human_decision`` : {"decision", "user", "comment"}."""
    e = html.escape
    parts: list[str] = []
    parts.append(f"<h1>Rapport d'analyse — Médicure AI</h1>")
    parts.append(
        f'<p class="muted">Analyse n° {e(result.analysis_id)} · {e(result.created_at)} · '
        f"empreinte image {e(result.image_sha256[:12])}…</p>"
    )
    if result.model_info.get("simulation"):
        parts.append('<div class="box bad"><b>MODE SIMULATION</b> — scores fictifs, aucune valeur médicale.</div>')
    parts.append(f'<div class="box warn">{e(DISCLAIMER)}</div>')

    # --- statut / incertitude d'abord
    parts.append("<h2>1. Statut et fiabilité</h2>")
    if result.status == "unsupported":
        parts.append('<div class="box bad">' + "<br>".join(e(m) for m in result.messages) + "</div>")
    else:
        cls = {"ok": "ok", "refused": "bad"}[result.status]
        parts.append(f'<div class="box {cls}">' + "<br>".join(e(m) for m in result.messages) + "</div>")
        if result.reliability:
            r = result.reliability
            comps = " · ".join(f"{e(k)} : {v:.0f}" for k, v in r.components.items())
            parts.append(
                f'<p><span class="idx">{r.index:.0f}/100</span> indice de fiabilité ({e(r.level)})<br>'
                f'<span class="muted">Composantes — {comps}</span></p>'
            )
        if result.decision:
            d = result.decision
            emoji = PRIORITY_EMOJI.get(d.priority, "")
            parts.append(f"<p><b>Priorité :</b> {emoji} {e(d.priority)}")
            if d.human_review_required:
                parts.append(" — <b>vérification humaine requise</b>")
            if d.second_opinion_required:
                parts.append(" — <b>second avis requis</b>")
            parts.append("</p>")

    # --- qualité
    if result.quality:
        q = result.quality
        parts.append("<h2>2. Qualité de l'image</h2>")
        parts.append(f"<p>Score qualité : {q.score:.0f}/100. Type d'examen : radiographie thoracique (confiance {result.exam.confidence:.2f}).</p>")
        if q.issues:
            parts.append("<ul>" + "".join(f"<li>{e(m)}</li>" for m in q.issues) + "</ul>")

    # --- scores (seulement si le système accepte de conclure)
    if result.scores_visible and result.targets:
        parts.append("<h2>3. Résultats par anomalie ciblée</h2>")
        parts.append("<table><tr><th>Anomalie</th><th>Score</th><th>Stabilité (écart-type)</th></tr>")
        for t in result.targets:
            parts.append(f"<tr><td>{e(t.label)}</td><td>{e(t.display)}</td><td>{t.std:.3f}</td></tr>")
        parts.append("</table>")
        parts.append("<h2>4. Explications</h2>")
        for t in result.targets:
            parts.append(f"<p>{e(result.explanations.get(t.name, ''))}</p>")
            if t.name in result.overlays:
                parts.append(_img_tag(result.overlays[t.name]))
        parts.append(
            '<p class="muted">La carte de chaleur est grossière (grille 7×7) et montre où le modèle a porté son '
            "attention, pas la cause médicale. Orientation supposée standard (côté droit du patient à gauche de l'image).</p>"
        )
    elif result.status == "refused":
        parts.append("<h2>3. Résultats</h2><p>Aucun score n'est présenté : le système refuse de conclure.</p>")

    # --- décision humaine (jamais pré-cochée)
    parts.append("<h2>Décision du professionnel de santé</h2>")
    if human_decision:
        parts.append(
            f"<p><b>{e(human_decision['decision'])}</b> par {e(human_decision.get('user', 'anonyme'))}"
            + (f" — {e(human_decision['comment'])}" if human_decision.get("comment") else "")
            + "</p>"
        )
    else:
        parts.append("<p>Aucune décision enregistrée.</p>")

    # --- traçabilité
    parts.append("<h2>Traçabilité</h2>")
    parts.append(
        f'<p class="muted">Modèle : {e(str(result.model_info.get("weights", "?")))} · '
        f"calibration : {e(result.calibration_info)} · temps de traitement : {result.elapsed_s:.1f} s.<br>"
        "Limites : radiographie thoracique de face uniquement, trois anomalies ciblées, biais démographiques "
        "et décalage de distribution non exclus.</p>"
    )

    body = "\n".join(parts)
    return f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Rapport Médicure AI</title><style>{_CSS}</style></head><body>{body}</body></html>'


def save_report(result, path: str | Path, human_decision: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, human_decision), encoding="utf-8")
    return path
