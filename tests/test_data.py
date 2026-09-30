import cv2
import numpy as np
import pandas as pd

from medguard.data import binarize, load_chexpert, load_nih, predict_paths, sample_by_patient
from medguard.mock import MockPredictor
from medguard.paths import find_file
from medguard.synthetic import fake_chest


def _write_png(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), fake_chest(256, seed))


def test_load_nih_parses_labels_and_finds_images(tmp_path):
    for i in range(3):
        _write_png(tmp_path / "images_001" / "images" / f"0000000{i}_000.png", i)
    pd.DataFrame(
        {
            "Image Index": ["00000000_000.png", "00000001_000.png", "00000002_000.png", "missing.png"],
            "Finding Labels": ["Effusion|Mass", "No Finding", "Cardiomegaly|Effusion", "Effusion"],
            "Patient ID": [1, 2, 3, 4],
            "Patient Age": [50, 60, 70, 40],
            "Patient Gender": ["M", "F", "M", "F"],
            "View Position": ["PA", "AP", "PA", "PA"],
        }
    ).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    df = load_nih(tmp_path)
    assert len(df) == 3  # l'image absente du disque est ignorée
    assert df["Effusion"].tolist() == [1.0, 0.0, 1.0] and df["Cardiomegaly"].tolist() == [0.0, 0.0, 1.0]


def test_load_chexpert_frontal_only_and_uncertain_values(tmp_path):
    rel = "CheXpert-v1.0-small/valid/patient00001/study1/view1_frontal.jpg"
    rel_lat = "CheXpert-v1.0-small/valid/patient00001/study1/view2_lateral.jpg"
    for r in (rel, rel_lat):
        (tmp_path / r).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / r), fake_chest(128))
    pd.DataFrame(
        {
            "Path": [rel, rel_lat],
            "Frontal/Lateral": ["Frontal", "Lateral"],
            "Pleural Effusion": [-1.0, 1.0],
            "Cardiomegaly": [1.0, 0.0],
            "Lung Opacity": [np.nan, 1.0],
        }
    ).to_csv(tmp_path / "valid.csv", index=False)
    df = load_chexpert(tmp_path)
    assert len(df) == 1 and df.loc[0, "patient_id"] == "00001"
    assert df.loc[0, "Effusion"] == -1.0
    assert np.isnan(binarize(df["Effusion"], "drop").iloc[0])
    assert binarize(df["Effusion"], "positive").iloc[0] == 1.0


def test_sample_by_patient_keeps_patients_whole():
    df = pd.DataFrame({"patient_id": np.repeat(np.arange(100), 5).astype(str), "x": range(500)})
    small = sample_by_patient(df, 100, seed=1)
    assert 100 <= len(small) <= 105
    assert (small["patient_id"].value_counts() == 5).all()


def test_predict_paths_batches(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        _write_png(p, i)
        paths.append(str(p))
    out = predict_paths(MockPredictor(), paths, batch_size=2, progress=False)
    assert out.shape == (5, 3)


def test_find_file_searches_given_dirs_then_raises(tmp_path):
    (tmp_path / "calibration.json").write_text("{}")
    assert find_file("calibration.json", [tmp_path]) == tmp_path / "calibration.json"
    try:
        find_file("nope.json", [tmp_path])
    except FileNotFoundError as e:
        assert "nope.json" in str(e)
    else:
        raise AssertionError("aurait dû lever FileNotFoundError")


def test_audit_log_path_is_overridable_via_env(monkeypatch, tmp_path):
    """MEDGUARD_AUDIT_LOG doit permettre de faire pointer le journal vers un stockage persistant."""
    monkeypatch.setenv("MEDGUARD_AUDIT_LOG", str(tmp_path / "custom" / "audit.jsonl"))
    import importlib

    import medguard.config as config_mod
    importlib.reload(config_mod)
    try:
        assert config_mod.AUDIT_LOG_PATH == tmp_path / "custom" / "audit.jsonl"
    finally:
        monkeypatch.delenv("MEDGUARD_AUDIT_LOG", raising=False)
        importlib.reload(config_mod)
