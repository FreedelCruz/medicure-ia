# Image de déploiement de Médicure AI — MedGuard.
# Conçue pour Hugging Face Spaces (SDK "docker"), mais fonctionne sur tout hébergeur de conteneurs.
FROM python:3.11-slim

# opencv-python-headless n'a pas besoin de libGL (c'est tout l'intérêt de la variante "headless") ;
# ces deux paquets système couvrent les rares cas où libjpeg/libpng manqueraient sur l'image slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY medguard ./medguard
COPY app ./app
COPY app.py .
# Artefacts déjà commités (calibration.json, thresholds.json...), s'ils existent : sans eux,
# l'application démarre quand même et affiche des scores "non calibrés" (jamais de faux pourcentage).
COPY artifacts ./artifacts

# Poids TorchXRayVision pré-téléchargés DANS L'IMAGE : le premier utilisateur n'attend pas un
# téléchargement, et le build échoue tout de suite si l'environnement torch/numpy est incohérent
# (le même contrôle que medguard/model.py fait au démarrage) plutôt qu'en production.
ENV MEDGUARD_WEIGHTS_CACHE=/app/.cache/torchxrayvision
RUN mkdir -p "$MEDGUARD_WEIGHTS_CACHE" \
    && python -c "from medguard.model import XRVPredictor; XRVPredictor(); print('poids en cache : OK')"

ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
CMD ["python", "app.py"]
