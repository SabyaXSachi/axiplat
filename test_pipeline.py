"""Unit tests for the improved AXIPLAT accuracy pipeline.

Run with the project venv:
    python test_pipeline.py

Covers: outlier rejection, face-quality gating math, EMA behavior,
DNN soft estimate sanity, /status fields, and session-bound token issuing.
"""

import os
import types

import numpy as np

import app as axiplat


def make_landmarks(n, eye_left_x, eye_right_x, nose_x, y=0.5):
    """Build a FakeMediaPipe landmark list with .x/.y in normalized coords."""
    class LM:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    lms = [LM(0.5, y) for _ in range(n)]  # neutral cloud centered in frame
    lms[axiplat.LM_LEFT_EYE_OUTER] = LM(eye_left_x, y)
    lms[axiplat.LM_RIGHT_EYE_OUTER] = LM(eye_right_x, y)
    lms[axiplat.LM_NOSE_TIP] = LM(nose_x, y)
    return types.SimpleNamespace(landmark=lms)


def spread_landmarks(lms, x0, x1, y0, y1):
    """Give the fake landmarks a bounding box [x0,x1]x[y0,y1] (normalized)."""
    n = len(lms.landmark)
    for i, lm in enumerate(lms.landmark):
        lm.x = x0 + (x1 - x0) * (i % 10) / 9.0
        lm.y = y0 + (y1 - y0) * (i // 10 % 10) / 9.0
    # Re-pin the three anatomy points used by the quality gate
    lms.landmark[axiplat.LM_LEFT_EYE_OUTER].x = x0
    lms.landmark[axiplat.LM_RIGHT_EYE_OUTER].x = x1
    lms.landmark[axiplat.LM_NOSE_TIP].x = (x0 + x1) / 2.0
    return lms


def test_reject_outliers_iqr():
    vals = [20.0, 21.0, 22.0, 23.0, 90.0]
    kept = axiplat.reject_outliers_iqr(vals)
    assert 90.0 not in kept, f"extreme outlier should be dropped, got {kept}"
    assert len(kept) == 4

    tight = [25.0, 25.1, 24.9, 25.05]
    assert axiplat.reject_outliers_iqr(tight) == tight, "clean data must be untouched"

    small = [10.0, 99.0, 11.0]
    assert axiplat.reject_outliers_iqr(small) == small, "small sets returned as-is"

    print("PASS reject_outliers_iqr")


def test_assess_face_quality():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    n = 468

    # Frontal, close face: eye corners at the box edges, nose centered
    lms = spread_landmarks(make_landmarks(n, 0, 0, 0), 0.30, 0.70, 0.30, 0.70)
    face, q = axiplat.assess_face_quality(frame, lms)
    assert face is not None
    assert q["size_ok"] is True, f"wide face should pass size check: {q}"
    assert q["pose_ok"] is True, f"centered nose should pass pose check: {q}"
    assert abs(q["yaw_ratio"]) < 0.01, f"frontal yaw should be ~0: {q}"

    # Far face: same shape, small box
    lms = spread_landmarks(make_landmarks(n, 0, 0, 0), 0.48, 0.52, 0.48, 0.52)
    _, q = axiplat.assess_face_quality(frame, lms)
    assert q["size_ok"] is False, f"tiny face should fail size check: {q}"

    # Turned face: nose pushed toward one side of the eye line
    lms = spread_landmarks(make_landmarks(n, 0, 0, 0), 0.30, 0.70, 0.30, 0.30)
    lms.landmark[axiplat.LM_NOSE_TIP].x = 0.30 + 0.02  # strongly off-center
    _, q = axiplat.assess_face_quality(frame, lms)
    assert q["pose_ok"] is False, f"off-center nose should fail pose check: {q}"
    assert q["yaw_ratio"] < -0.25, f"yaw ratio should be strongly negative: {q}"

    print("PASS assess_face_quality")


def test_ema_update():
    assert axiplat.ema_update(None, 30.0) == 30.0, "first value seeds the EMA"
    v = axiplat.ema_update(30.0, 20.0, alpha=0.3)
    assert abs(v - (30.0 * 0.7 + 20.0 * 0.3)) < 1e-9
    print("PASS ema_update")


def test_dnn_soft_estimate():
    if not axiplat.USE_DNN:
        print("SKIP dnn_soft_estimate (no DNN model files)")
        return
    face = np.full((100, 100, 3), 120, dtype=np.uint8)
    age = axiplat.soft_expected_age_from_dnn(face)
    assert age is not None and np.isfinite(age) and age >= 0, f"bad estimate: {age}"
    # Identity calibration: output should be near the bucket midpoints (0-100)
    assert age <= 100, f"estimate out of range: {age}"
    print(f"PASS dnn_soft_estimate (blank-face estimate {age:.1f} yrs)")


def test_status_fields():
    client = axiplat.app.test_client()
    resp = client.get("/status")
    assert resp.status_code == 200
    payload = resp.get_json()
    for key in ("stream_active", "session_id", "face_detected", "ema",
                "valid_samples", "samples_needed", "elapsed", "max_seconds",
                "threshold", "calibration", "final_median_age", "final_group",
                "access_allowed", "camera_error"):
        assert key in payload, f"/status missing field: {key}"
    assert payload["threshold"] == axiplat.MIN_AGE_THRESHOLD
    print("PASS status_fields")


def test_session_bound_token():
    # Simulate a completed verification session
    axiplat.access_allowed = True
    axiplat.current_session = {"id": "sess-abc", "redeemed": False}

    client = axiplat.app.test_client()

    # Wrong sid -> no token
    resp = client.get("/insta?sid=wrong")
    html = resp.get_data(as_text=True)
    assert "age_gate_token" not in html, "token must not be issued for a wrong sid"

    # No sid -> no token
    resp = client.get("/insta")
    html = resp.get_data(as_text=True)
    assert "age_gate_token" not in html, "token must not be issued without sid"

    # Correct sid -> token issued once, then session is redeemed
    resp = client.get("/insta?sid=sess-abc")
    html = resp.get_data(as_text=True)
    assert "age_gate_token" in html, "correct sid should receive a token"
    assert axiplat.current_session["redeemed"] is True

    # Replay of the same sid -> no second token
    resp = client.get("/insta?sid=sess-abc")
    html = resp.get_data(as_text=True)
    assert "age_gate_token" not in html, "redeemed session must not re-issue"

    # Deny case: allowed flag off -> never a token
    axiplat.access_allowed = False
    axiplat.current_session = {"id": "sess-xyz", "redeemed": False}
    resp = client.get("/insta?sid=sess-xyz")
    html = resp.get_data(as_text=True)
    assert "age_gate_token" not in html, "token must not issue when access denied"

    print("PASS session_bound_token")


def test_calibration_fit():
    # Empty -> identity
    assert axiplat.fit_calibration([]) == (1.0, 0.0)
    # One point -> offset-only
    scale, offset = axiplat.fit_calibration([(28.47, 21.0)])
    assert scale == 1.0 and abs(offset - (21.0 - 28.47)) < 1e-9
    # Two distinct points -> least squares: (10->20, 30->40) is exact y = x + 10
    scale, offset = axiplat.fit_calibration([(10.0, 20.0), (30.0, 40.0)])
    assert abs(scale - 1.0) < 1e-9 and abs(offset - 10.0) < 1e-9
    # Degenerate: same measured values -> offset-only fallback, no crash
    scale, offset = axiplat.fit_calibration([(25.0, 20.0), (25.0, 30.0)])
    assert scale == 1.0 and abs(offset - 0.0) < 1e-9
    # Degenerate: same true values -> offset-only (slope would collapse to 0)
    scale, offset = axiplat.fit_calibration([(20.0, 21.0), (30.0, 21.0)])
    assert scale == 1.0 and abs(offset - (21.0 - 25.0)) < 1e-9
    print("PASS calibration_fit")


def test_calibration_model_guard():
    # Points must only be used for the estimator they were measured against
    caffe = axiplat.load_calibration_points("caffe-dnn")
    ga = axiplat.load_calibration_points("genderage-onnx")
    none = axiplat.load_calibration_points("does-not-exist")
    assert len(caffe) == 2, f"expected 2 caffe-dnn points, got {len(caffe)}"
    assert len(ga) == 2, f"expected 2 genderage-onnx points, got {len(ga)}"
    assert none == []
    # Two same-true points -> offset-only fit: 21 - mean(measured)
    ga_measured = [m for m, _ in ga]
    expected_offset = 21.0 - (sum(ga_measured) / len(ga_measured))
    env_backup = {k: os.environ.pop(k, None) for k in
                  ("AGE_CALIBRATION_SCALE", "AGE_CALIBRATION_OFFSET")}
    try:
        scale, offset = axiplat.load_effective_calibration("genderage-onnx")
        assert scale == 1.0 and abs(offset - expected_offset) < 0.01, (scale, offset)
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v
    print("PASS calibration_model_guard")


def test_active_estimator():
    # On this machine genderage.onnx is present, so it must be the primary
    assert axiplat.PRIMARY_MODEL_NAME == "genderage-onnx", axiplat.PRIMARY_MODEL_NAME
    assert axiplat.genderage_session is not None
    # And the effective calibration matches the fitted offset
    ga = axiplat.load_calibration_points("genderage-onnx")
    expected_offset = 21.0 - (sum(m for m, _ in ga) / len(ga))
    assert abs(axiplat.CALIBRATION_OFFSET - expected_offset) < 0.01, axiplat.CALIBRATION_OFFSET
    print("PASS active_estimator")


def test_decide_access():
    # No samples -> sample error message
    median, group, allowed, err = axiplat.decide_access([], 0)
    assert median is None and not allowed and "samples" in err

    # Enough samples + median 21 >= 14 -> granted, no error
    vals = [21.0, 21.1, 20.9, 21.0, 21.2, 21.0]
    median, group, allowed, err = axiplat.decide_access(vals, 6)
    assert abs(median - 21.0) < 1e-9 and allowed and err is None and group

    # Child median 10 -> denied cleanly, not an error message
    vals_child = [10.0, 10.1, 9.9, 10.0, 10.2, 10.0]
    median, group, allowed, err = axiplat.decide_access(vals_child, 6)
    assert abs(median - 10.0) < 1e-9 and not allowed and err is None
    print("PASS decide_access")


def main():
    tests = [
        test_reject_outliers_iqr,
        test_assess_face_quality,
        test_ema_update,
        test_dnn_soft_estimate,
        test_calibration_fit,
        test_calibration_model_guard,
        test_active_estimator,
        test_decide_access,
        test_status_fields,
        test_session_bound_token,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    total = len(tests)
    print(f"\n{total - failed}/{total} tests passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
