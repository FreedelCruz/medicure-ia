"""Fabrique un FAUX dataset au format NIH ChestX-ray14 (images synthétiques + CSV).

Sert à faire un "smoke test" des notebooks sans télécharger 40 Go de données :

    python scripts/make_fake_dataset.py --out /tmp/fake_nih --patients 150
    MEDGUARD_MOCK=1 MEDGUARD_DATA=/tmp/fake_nih jupyter nbconvert --execute notebooks/01_eda.ipynb ...

Les étiquettes sont corrélées à la luminosité de certaines zones (pour que le prédicteur
SIMULÉ obtienne une AUROC non triviale) et une partie des images est volontairement dégradée
(floues / sombres) pour exercer le contrôle qualité. Rien de tout cela n'a de valeur médicale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medguard.synthetic import fake_chest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=150)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    img_dir = out / "images_001" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n = a.size
    for pid in range(1, a.patients + 1):
        for k in range(int(rng.integers(1, 4))):
            name = f"{pid:08d}_{k:03d}.png"
            img = fake_chest(n, seed=int(rng.integers(0, 10**6))).astype(np.float32)
            effusion = rng.random() < 0.25
            cardio = rng.random() < 0.20
            if effusion:
                img[int(n * 0.68) :, int(n * 0.15) : int(n * 0.85)] += 45
            if cardio:
                img[int(n * 0.38) : int(n * 0.82), int(n * 0.25) : int(n * 0.75)] += 30
            img = np.clip(img, 0, 255).astype(np.uint8)
            r = rng.random()
            if r < 0.06:
                img = cv2.GaussianBlur(img, (0, 0), 14)  # image floue
            elif r < 0.10:
                img = (img.astype(np.float32) * 0.25).astype(np.uint8)  # image sombre
            cv2.imwrite(str(img_dir / name), img)
            labels = [l for l, on in (("Effusion", effusion), ("Cardiomegaly", cardio)) if on] or ["No Finding"]
            rows.append(
                {
                    "Image Index": name,
                    "Finding Labels": "|".join(labels),
                    "Follow-up #": k,
                    "Patient ID": pid,
                    "Patient Age": int(rng.integers(20, 90)),
                    "Patient Gender": str(rng.choice(["M", "F"])),
                    "View Position": str(rng.choice(["PA", "AP"], p=[0.6, 0.4])),
                }
            )
    pd.DataFrame(rows).to_csv(out / "Data_Entry_2017.csv", index=False)
    print(f"{len(rows)} images synthétiques écrites dans {out}")


if __name__ == "__main__":
    main()
