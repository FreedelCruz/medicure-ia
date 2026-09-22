# demo_assets

Déposez ici 5 à 8 images pour la démonstration (elles ne sont pas versionnées, cf. `.gitignore`).
Utilisez uniquement des images **publiques et anonymisées** (NIH ChestX-ray14, CheXpert) dont vous
respectez les conditions d'utilisation.

Sélection conseillée (voir `docs/DEMO.md`) :

1. un cas clairement anormal, bien détecté (épanchement ou cardiomégalie marqués) ;
2. un cas clairement normal ;
3. une image floue ou sombre (dégradez-en une avec `medguard.preprocess.apply_perturbation`) ;
4. un cas ambigu (score proche de l'indécision, trouvé avec le notebook 04) ;
5. une image qui n'est pas une radio thoracique (photo quelconque).

Repérez à l'avance les cas 1 et 4 avec le vrai modèle : ne les découvrez pas devant le jury.
