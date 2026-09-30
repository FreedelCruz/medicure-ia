# Mettre Médicure AI en ligne

**Recommandé : Hugging Face Spaces, SDK Docker.** Gratuit (CPU), simple, et l'image Docker
garantit que l'environnement qui plante localement (conflit NumPy/PyTorch, cf. votre incident)
ne peut pas se reproduire en production : le build échoue tout de suite si torch et NumPy sont
incompatibles (voir `Dockerfile`), au lieu de planter au premier clic d'un visiteur.

## 1. Créer le Space

1. https://huggingface.co/new-space
2. **SDK** : `Docker`. **Visibilité** : Public (ou Private si vous voulez la garder cachée jusqu'au
   jugement, cf. §4). **Matériel** : CPU basic (gratuit) suffit pour une démo.
3. Un Space = un dépôt Git séparé (`https://huggingface.co/spaces/<vous>/<nom>`). Clonez-le.

## 2. Y mettre le code

Copiez dans le clone du Space : `medguard/`, `app/`, `app.py`, `Dockerfile`, `.dockerignore`,
`requirements-space.txt`, et `artifacts/` (avec `calibration.json` etc. si vous les avez déjà
générés — sinon l'app tourne quand même, en affichant des scores non calibrés).

Le **README.md du Space** (pas celui du dépôt principal) doit commencer par ce bloc, sinon HF ne
sait pas comment construire le Space :

```yaml
---
title: Médicure AI — MedGuard
emoji: 🩺
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---
```

Puis :

```bash
git add -A
git commit -m "Déploiement MedGuard"
git push
```

Le build prend plusieurs minutes (torch + téléchargement des poids pendant le build). Suivez-le
dans l'onglet **Logs** du Space.

## 3. Variables d'environnement (onglet *Settings → Variables and secrets* du Space)

| Variable | Effet |
|---|---|
| `MEDGUARD_MOCK=1` | démo sans le vrai modèle, pour tester le déploiement rapidement |
| `MEDGUARD_WEIGHTS` | change les poids TorchXRayVision (défaut : `densenet121-res224-chex`) |
| `MEDGUARD_AUTH_USER` / `MEDGUARD_AUTH_PASS` | protège l'accès par mot de passe (voir §4) |
| `MEDGUARD_AUDIT_LOG` | chemin du journal d'audit, ex. `/data/audit_log.jsonl` avec le stockage persistant (§5) |

## 4. Le garder caché jusqu'au jugement

Un Space Docker public est accessible dès que le build réussit. Pour ne pas être découvert avant
l'heure : mettez le Space en **Private**, ou définissez `MEDGUARD_AUTH_USER`/`MEDGUARD_AUTH_PASS`
et donnez ces identifiants au jury.

## 5. Ce qui ne survit PAS à un redémarrage (important)

Sans l'option payante **Persistent Storage** (*Settings → Persistent storage*), le disque du Space
est réinitialisé à chaque redéploiement ou remise en veille. Concrètement :

- le **journal d'audit** (`artifacts/audit_log.jsonl`) repart à zéro — acceptable pour une démo de
  hackathon, mais pas pour un usage réel ;
- les **rapports HTML** téléchargés depuis l'interface restent accessibles tant que le conteneur
  tourne, pas après.

Pour le journal d'audit, si vous voulez qu'il survive : activez le stockage persistant (montré sur
`/data`) et mettez `MEDGUARD_AUDIT_LOG=/data/audit_log.jsonl`.

## 6. Vitesse

C'est du CPU : comptez 1 à 3 secondes par analyse (6 passes du modèle : image originale + 5
perturbations pour mesurer la stabilité). Correct pour une démo, pas pour un usage à grande
échelle. `demo.queue(...)` (déjà dans `app.py`) met les visiteurs simultanés en file plutôt que de
saturer le serveur. Pour aller plus vite : un Space GPU payant, sans changer une ligne de code
(torch bascule sur CUDA automatiquement, cf. `XRVPredictor.__init__`).

## 7. Tester l'image en local avant de pousser

```bash
docker build -t medguard .
docker run --rm -p 7860:7860 -e MEDGUARD_MOCK=1 medguard   # rapide, sans torch
docker run --rm -p 7860:7860 medguard                       # avec le vrai modèle
```

Ouvrez http://localhost:7860.

## 8. Alternatives à Hugging Face Spaces

- **Space Gradio (sans Dockerfile)** : plus simple à mettre en place, mais moins de contrôle sur
  les paquets système. À essayer si vous voulez éviter Docker ; basculez vers le SDK Docker si
  l'import d'OpenCV échoue au démarrage.
- **Render, Railway, Fly.io** : utilisent directement ce `Dockerfile`. Utile si vous voulez un
  domaine personnalisé ou l'API FastAPI (`app/api.py`) en plus de l'interface — non couvert ici,
  seule l'interface Gradio est déployée par ce guide.

## Rappel

Le bandeau « outil non diagnostique, l'IA observe, le professionnel de santé décide » reste
affiché sur toute la page. Gardez-le : c'est la seule protection dont dispose un visiteur public
qui n'a pas lu le cahier des charges.
