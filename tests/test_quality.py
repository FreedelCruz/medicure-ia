from medguard.exam_check import check_exam_type
from medguard.quality import assess_quality


def test_clean_image_is_usable(chest_img):
    q = assess_quality(chest_img)
    assert q.usable
    assert q.score > 80
    assert q.fatal_issues == []


def test_blurry_image_is_rejected_with_clear_message(blurry_img):
    q = assess_quality(blurry_img)
    assert not q.usable
    assert any("floue" in m for m in q.fatal_issues)


def test_dark_image_is_rejected(dark_img):
    q = assess_quality(dark_img)
    assert not q.usable
    assert any("sombre" in m or "Contraste" in m for m in q.fatal_issues)


def test_low_resolution_is_rejected(lowres_img):
    q = assess_quality(lowres_img)
    assert not q.usable
    assert any("Résolution" in m for m in q.fatal_issues)


def test_quality_score_is_lower_for_bad_images(chest_img, blurry_img):
    assert assess_quality(blurry_img).score < assess_quality(chest_img).score


def test_chest_like_image_passes_exam_check(chest_img):
    e = check_exam_type(chest_img)
    assert e.is_chest_xray
    assert e.confidence > 0.8


def test_color_photo_is_not_a_chest_xray(color_photo_img):
    e = check_exam_type(color_photo_img)
    assert not e.is_chest_xray
    assert e.reasons
