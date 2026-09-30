# Logique de décision de MedGuard

Ce document fixe les règles implémentées dans `medguard/reliability.py`, et signale où elles
**s'écartent** du cahier des charges initial (et pourquoi). À relire avant l'oral : ce sont les
questions qu'un jury technique posera.

## Deux axes, pas un seul

| Axe | Question posée | Calcul |
|---|---|---|
| **Suspicion clinique** | « Y a-t-il probablement une anomalie ? » | probabilité **calibrée** maximale parmi les 3 cibles |
| **Fiabilité** (0-100) | « Peut-on faire confiance à ce résultat ? » | qualité d'image, netteté de la décision, stabilité aux perturbations |

La **priorité** vient de la suspicion. La **fiabilité** décide si le système accepte de conclure.

### Indice de fiabilité

Moyenne **géométrique** pondérée de trois composantes (chacune dans [0, 1]) :

- `qualité` : score du contrôle qualité (flou, contraste, luminosité, résolution, surexposition) ;
- `netteté de la décision` : `|2p − 1|` par pathologie (0 = coup de pile ou face, 1 = certitude), agrégée par `0,5·min + 0,5·moyenne` ;
- `stabilité` : `1 − écart-type max / 0,15` des scores calibrés sous 5 perturbations mineures (luminosité ±5 %, bruit, JPEG 85, recadrage 3 %).

Poids par défaut : 0,35 / 0,40 / 0,25 (`config.ReliabilityWeights`). Une image jugée **inexploitable** plafonne l'indice sous le seuil de refus.

### Table de décision

| Fiabilité | Suspicion | Statut | Priorité affichée | Drapeaux |
|---|---|---|---|---|
| ≥ 80 | Faible | analyse | 🟢 Faible | – |
| ≥ 80 | Intermédiaire | analyse | 🟠 Intermédiaire | – |
| ≥ 80 | Élevée | analyse | 🔴 Élevée | second avis requis |
| 50 – 79 | toutes | analyse | = suspicion | vigilance |
| < 50 | toutes | **refus de conclure** | max(suspicion, `refused_min_priority`) | vérification humaine + second avis |

Aucun score ni carte de chaleur n'est présenté quand le système refuse de conclure.

## Écarts avec le cahier des charges (à assumer)

1. **Priorité ≠ inverse de la confiance.** La section 5.4 disait « confiance ≥ 80 ⇒ priorité faible ». Un épanchement massif détecté avec certitude aurait été « peu prioritaire ». Corrigé : la priorité suit la suspicion ; la fiabilité pilote le refus. Test : `test_confident_severe_finding_is_HIGH_priority_not_low`.
2. **Un cas incertain n'est jamais neutre** (il ne peut pas descendre sous `refused_min_priority`, « Intermédiaire » par défaut). Pour reproduire à la lettre le cahier des charges, mettre `"Élevée"` dans `thresholds.json`.
3. **Moyenne géométrique** plutôt qu'arithmétique : avec une moyenne arithmétique, une image nette et stable pouvait masquer un score à 50/50 (indice ≈ 60 au lieu d'un refus). Test : `test_ambiguous_case_refuses_to_conclude_and_is_never_neutral`.
4. **Le modèle n'est pas appelé** sur une image hors périmètre ou de qualité insuffisante.
5. **Pourcentage interdit sans calibration** : une cible non calibrée est affichée « score brut, non calibré ».
6. **« 100 % des cas prioritaires détectés »** remplacé par une sensibilité **avec intervalle de confiance** (notebook 03) et la courbe risque–couverture (notebook 04).

## Ce qui est une valeur de départ (à recaler sur les données)

`reliability_refuse = 50`, `reliability_high = 80`, `suspicion_high = 0,60`, `suspicion_medium = 0,30`, tous les seuils qualité, `stability_max_std = 0,15`, les poids de l'indice. Le notebook 01 cale les seuils qualité, le notebook 04 règle le seuil de refus sur `dev` et le rapporte sur `test`.

## Limites à dire à voix haute

- Le poids des composantes de l'indice est choisi à la main ; la courbe risque–couverture (notebook 04) dit s'il « marche », pas s'il est optimal.
- Les scores TorchXRayVision sont normalisés autour d'un point de fonctionnement : sans calibration ils ne sont pas des probabilités.
- La stabilité mesurée n'explore que des perturbations mineures : elle n'attrape pas un décalage de distribution (autre hôpital, autre appareil).
- Grad-CAM est grossier (grille 7×7) et n'explique pas la cause médicale.
- Le détecteur de type d'examen est une heuristique (couleur, proportions, symétrie).
