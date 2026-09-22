"""Interface Gradio de Médicure AI.

    python -m app.gradio_app --mock        # sans torch ni poids (interface seulement)
    python -m app.gradio_app               # vrai modèle (TorchXRayVision)
    python -m app.gradio_app --share       # lien public temporaire (démo à distance)

Principes d'interface (cahier des charges, section 10.1) :
* l'incertitude et la fiabilité sont affichées AVANT les scores ;
* aucune décision n'est pré-cochée ;
* la validation humaine est journalisée, jamais réutilisée pour réentraîner.
"""
from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from app.formatting import DISCLAIMER, explanations_md, quality_md, scores_md, status_md
from medguard.audit import VALID_DECISIONS, log_analysis, log_human_decision
from medguard.pipeline import MedGuardPipeline, build_default_pipeline
from medguard.report import save_report

REPORT_DIR = Path(tempfile.gettempdir()) / "medguard_reports"


def pipeline_label(result, name):
    return next((t.label for t in result.targets if t.name == name), name)


def safe_analyze(pipeline: MedGuardPipeline, image, user):
    """Analyse + mise en forme. Toute erreur devient un message lisible (pas un « Erreur » muet)."""
    if image is None:
        return ("⬆️ Chargez d'abord une radiographie (JPG ou PNG).", "", "", "", [], None, None)
    try:
        result = pipeline.analyze(image)
        log_analysis(result, user=(user or "anonyme").strip() or "anonyme")
        report_path = save_report(result, REPORT_DIR / f"rapport_{result.analysis_id}.html")
        gallery = [(img, pipeline_label(result, name)) for name, img in result.overlays.items()]
        return (
            status_md(result),
            quality_md(result),
            scores_md(result),
            explanations_md(result),
            gallery,
            str(report_path),
            result.analysis_id,
        )
    except Exception as exc:
        traceback.print_exc()  # détail complet dans le terminal
        msg = (
            "## ❌ Erreur technique — l'analyse n'a pas pu être réalisée\n\n"
            f"`{type(exc).__name__}: {exc}`\n\nLe détail complet est affiché dans le terminal."
        )
        return (msg, "", "", "", [], None, None)


def build_ui(pipeline: MedGuardPipeline) -> gr.Blocks:
    simulation = bool(pipeline.predictor.info.get("simulation"))

    def analyze(image, user):
        return safe_analyze(pipeline, image, user)

    def record(analysis_id, decision, comment, user):
        if not analysis_id:
            return "Aucune analyse à valider."
        if not decision:
            return "Choisissez une décision (rien n'est pré-sélectionné)."
        log_human_decision(analysis_id, decision, user=(user or "anonyme").strip() or "anonyme", comment=comment or "")
        return f"✔️ Décision « {decision} » enregistrée dans le journal d'audit (aucun réentraînement automatique)."

    with gr.Blocks(title="Médicure AI — MedGuard") as demo:
        gr.Markdown("# 🩺 Médicure AI — MedGuard\n" + DISCLAIMER)
        if simulation:
            gr.Markdown("### 🧪 MODE SIMULATION — les scores sont fictifs. Ne pas utiliser pour la démonstration finale.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Radiographie thoracique (face)", image_mode="RGB")
                user = gr.Textbox(label="Utilisateur (pour le journal d'audit)", placeholder="ex. dr_dupont")
                run = gr.Button("Analyser", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown()
                quality = gr.Markdown()
        gr.Markdown("### Résultats")
        scores = gr.Markdown()
        explanations = gr.Markdown()
        gallery = gr.Gallery(label="Zones d'attention du modèle (Grad-CAM)", columns=3, height="auto")
        report = gr.File(label="Rapport téléchargeable (HTML)")
        analysis_id = gr.State(None)

        gr.Markdown("### Décision du professionnel de santé")
        decision = gr.Radio(list(VALID_DECISIONS), value=None, label="Votre décision (aucune n'est pré-sélectionnée)")
        comment = gr.Textbox(label="Commentaire (facultatif)", lines=2)
        save = gr.Button("Enregistrer la décision")
        saved = gr.Markdown()

        run.click(analyze, [image, user], [status, quality, scores, explanations, gallery, report, analysis_id])
        save.click(record, [analysis_id, decision, comment, user], [saved])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Interface Médicure AI")
    ap.add_argument("--mock", action="store_true", help="prédicteur simulé (sans torch ni poids)")
    ap.add_argument("--weights", default=None, help="poids TorchXRayVision (défaut : config.MODEL_WEIGHTS)")
    ap.add_argument("--share", action="store_true", help="lien public temporaire")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--auth", default=None, help="utilisateur:motdepasse (protège --share)")
    a = ap.parse_args()
    demo = build_ui(build_default_pipeline(weights=a.weights, mock=a.mock))
    demo.queue(max_size=20, default_concurrency_limit=2)
    auth = tuple(a.auth.split(":", 1)) if a.auth else None
    demo.launch(share=a.share, server_port=a.port, auth=auth, max_file_size="15mb")


if __name__ == "__main__":
    main()
