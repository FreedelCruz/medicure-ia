# Démonstration devant le jury

## Avant

- Démo **locale sur CPU** : pas de dépendance au réseau ni au GPU Kaggle.
- Lancez `python -m app.gradio_app` (le vrai modèle, **pas** `--mock`) et vérifiez l'absence de bannière « MODE SIMULATION ».
- Préchargez le modèle (une première analyse à blanc) : le premier appel est plus lent.
- Repérez vos 5 images (voir `demo_assets/README.md`) **avec le vrai modèle** et notez leurs résultats.
- Enregistrez une **vidéo de secours** de la démo complète.
- Rapport HTML de chaque cas prêt à ouvrir.

## Script de 3 minutes

| Temps | Action | Message |
|---|---|---|
| 0:00 | Problème en une phrase | « Une IA qui donne un chiffre sans dire quand elle doute est dangereuse. Nous avons construit celle qui sait dire "je ne sais pas". » |
| 0:20 | **Cas 1 : anomalie nette** | Fiabilité haute, priorité élevée, carte Grad-CAM, phrase prudente. « L'IA observe. » |
| 1:00 | **Cas 2 : image floue** | Rejet immédiat *avant* le modèle, message clair. « Elle refuse d'analyser une image qu'elle ne peut pas lire. » |
| 1:25 | **Cas 3 : cas ambigu** | Refus de conclure, aucun score, vérification humaine requise. « Elle sait dire "je ne sais pas". » |
| 1:50 | **Cas 4 : photo hors périmètre** | « Hors périmètre », pas de faux diagnostic. |
| 2:05 | **La preuve** : courbe risque–couverture | « En refusant X % des cas, l'exactitude monte de A % à B % (IC 95 %). Un ordre aléatoire n'améliore rien. » |
| 2:35 | Validation humaine + rapport | Aucune décision pré-cochée, journal d'audit, aucun réentraînement automatique. « Le professionnel décide. » |
| 2:50 | Limites (une phrase) | Trois anomalies, radio de face, pas un dispositif médical, biais à évaluer. |

## Questions probables

- **« Pourquoi pas un modèle plus gros ? »** Le problème n'est pas la précision brute mais la confiance que l'on peut accorder à un résultat. Le modèle est interchangeable, la couche de confiance est le produit.
- **« Vos scores sont-ils des probabilités ? »** Après calibration sur un split dédié, mesurés sur un split test (ECE avant/après, notebook 03). Sans calibration, l'application n'affiche pas de pourcentage.
- **« Fuite de données ? »** Découpage par patient, poids et données externes l'un à l'autre (docs/KAGGLE.md, §4), test jamais utilisé pour régler quoi que ce soit.
- **« Une IA sûre d'elle sur un cas grave donne-t-elle une priorité faible ? »** Non : la priorité suit la suspicion clinique, la fiabilité pilote le refus (docs/DECISION_LOGIC.md).
- **« Votre indice de fiabilité marche-t-il ? »** Courbe risque–couverture et ablation par composante (notebook 04), robustesse aux dégradations (notebook 05).
- **« Et sur d'autres hôpitaux ? »** On ne le sait pas ; évaluation externe partielle seulement. C'est une limite, pas un point que l'on cache.
- **« Biais démographiques ? »** Distribution âge/sexe/incidence documentée (notebook 01), performances par sous-groupe non encore analysées.
