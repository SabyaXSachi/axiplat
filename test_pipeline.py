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


