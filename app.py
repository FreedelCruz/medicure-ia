"""Point d'entrée pour un déploiement en conteneur (Hugging Face Spaces, Docker...).

Contrairement à ``app/gradio_app.py`` (pensé pour un lancement local en ligne de commande,
avec des arguments ``--mock``/``--share``), ce fichier se pilote par variables d'environnement,
ce qui est la convention pour un conteneur :

    MEDGUARD_MOCK=1                       lance en mode simulation (sans torch ni poids)
    MEDGUARD_WEIGHTS=densenet121-res224-nih   change les poids TorchXRayVision
    MEDGUARD_AUTH_USER / MEDGUARD_AUTH_PASS   protège l'accès par mot de passe (recommandé
                                               pendant un jugement de hackathon, pour ne pas
                                               être découvert avant l'heure)
    PORT                                   port d'écoute (7860 par défaut, attendu par HF Spaces)
"""
from __future__ import annotations

import os

from app.gradio_app import build_ui
from medguard.pipeline import build_default_pipeline


def main() -> None:
    mock = os.environ.get("MEDGUARD_MOCK") == "1"
    weights = os.environ.get("MEDGUARD_WEIGHTS") or None
    pipeline = build_default_pipeline(weights=weights, mock=mock)

    demo = build_ui(pipeline)
    demo.queue(max_size=20, default_concurrency_limit=2)  # borne l'attente sur un serveur CPU public

    user = os.environ.get("MEDGUARD_AUTH_USER")
    password = os.environ.get("MEDGUARD_AUTH_PASS")
    auth = (user, password) if user and password else None

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        auth=auth,
        max_file_size="15mb",  # protège la mémoire/le CPU d'un serveur public contre les gros envois
    )


if __name__ == "__main__":
    main()
