import os
import time
import json
import secrets
import logging
from threading import Lock

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS

import mediapipe as mp

# TensorFlow regression model loads only when the dependency is present
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("axiplat")

app = Flask(__name__)
CORS(app)  # Allow extension to call API

# Base configuration values
MODEL_DIR = os.path.join(os.getcwd(), "model")
AGE_PROTO = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_CAFFE = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_LIST = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
# Numeric midpoint for each group (used to convert soft outputs to numeric age)
AGE_MIDPOINTS = [ (int(a.split('(')[1].split('-')[0]) + int(a.split('-')[1].split(')')[0]))/2.0 for a in AGE_LIST ]

# Sampling cadence and smoothing behavior
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = float(os.environ.get("AGE_MAX_SECONDS", "20"))
STABLE_SECONDS = 3
STABILITY_THRESH = 1.0
EMA_ALPHA = float(os.environ.get("AGE_EMA_ALPHA", "0.3"))
# Decision waits for this many quality-passing samples (prevents acting on
# a couple of noisy frames when the face keeps dropping out)
MIN_VALID_SAMPLES = int(os.environ.get("AGE_MIN_SAMPLES", "6"))

# Access decision threshold (minimum estimated age to grant access)
MIN_AGE_THRESHOLD = float(os.environ.get("AGE_MIN_THRESHOLD", "14.0"))

# Weighting for each model contribution (np.average normalizes over the
# models that are actually available, so a DNN-only setup gets weight 1.0)
DNN_WEIGHT = 0.6
CNN_WEIGHT = 0.4
CAMERA_INDEX = 0

# Adaptive normalization.
# Default OFF: the DNN was trained with a fixed mean subtraction, and
# deviating from it has never been validated against ground truth.
DEFAULT_BLOB_MEAN = (
    78.4263377603,
    87.7689143744,
    114.895847746
)
DYNAMIC_MEAN_BLEND = float(os.environ.get("AGE_DYNAMIC_MEAN_BLEND", "0.0"))

# Reference-point calibration: calibrated = scale * raw + offset.
# Fitted from calibration.json reference points for the active estimator
# (see calibrate.py); env vars override the file. Identity when neither
# provides anything.
CALIBRATION_FILE = "calibration.json"
CALIBRATION_SCALE_MIN = 0.3
CALIBRATION_SCALE_MAX = 2.0

# Face quality gating: reject samples when the face is too far away or the
# head is turned too far sideways (both reliably distort age estimates).
MIN_FACE_WIDTH_RATIO = float(os.environ.get("AGE_MIN_FACE_RATIO", "0.15"))
MAX_YAW_RATIO = float(os.environ.get("AGE_MAX_YAW_RATIO", "0.25"))

# Model loading
USE_DNN = False
age_net = None
if os.path.isfile(AGE_PROTO) and os.path.isfile(AGE_CAFFE):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_CAFFE)
        USE_DNN = True
        logger.info("Loaded OpenCV DNN age model.")
    except Exception:
        logger.exception("Failed to load DNN age model")

age_cnn_model = None
if TF_AVAILABLE:
    cnn_path = os.path.join(MODEL_DIR, "cnn_age_model.h5")
    if os.path.isfile(cnn_path):
        try:
            age_cnn_model = load_model(cnn_path)
            logger.info("Loaded CNN regression model: %s", cnn_path)
        except Exception:
            logger.exception("Failed to load CNN")
    else:
        logger.info("No CNN model found; continuing without regression CNN.")
else:
    logger.info("TensorFlow not available; CNN regression disabled.")

# insightface genderage.onnx (buffalo_l): modern age+gender model, the primary
# estimator when present. The old Caffe DNN stays as the fallback. Age comes
# from output element [2] * 100 - verified empirically against a known-age
# face in benchmark_models.py (input is a 96x96 blob from a 112x112
# ArcFace-aligned crop, swapRB=True).
GENDERAGE_ONNX = os.path.join(MODEL_DIR, "genderage.onnx")
genderage_session = None
GENDERAGE_INPUT_SIZE = (96, 96)
if os.path.isfile(GENDERAGE_ONNX):
    try:
        import onnxruntime as ort
        genderage_session = ort.InferenceSession(
            GENDERAGE_ONNX, providers=["CPUExecutionProvider"])
        _shape = genderage_session.get_inputs()[0].shape
        GENDERAGE_INPUT_SIZE = (int(_shape[3]), int(_shape[2]))
        logger.info("Loaded genderage ONNX age model (input %dx%d).", *GENDERAGE_INPUT_SIZE)
    except Exception:
        logger.exception("Failed to load genderage ONNX model (onnxruntime installed?)")
else:
    logger.info("No genderage.onnx found; Caffe DNN remains the estimator.")

# Face landmark detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Shared state between streaming and endpoints
lock = Lock()

latest_ema_age = None
sampled_ages = []
final_median_age = None
final_age_group = None
access_allowed = False
stream_active = False
camera_error = None
face_detected_now = False
stream_started_at = None

# Current verification session: a token is only issued to the browser that
# actually completed (or is viewing) this camera session, not to anyone
# who happens to open /insta after somebody else verified.
current_session = None  # {"id": str, "redeemed": bool}
SESSION_LOG_FILE = "session_log.json"

# Verification tokens stored in memory (persisted to json between runs)
verification_tokens = {}  # token -> timestamp
VERIFICATION_TOKEN_FILE = "verification_tokens.json"
VERIFICATION_EXPIRY_HOURS = float(os.environ.get("AGE_TOKEN_HOURS", "24"))

# Default fallback passcode (override via env)
PASSCODE = os.environ.get("AGE_GATE_PASSCODE", "admin123")
REMOTE_SHUTDOWN_ENABLED = os.environ.get("AXIPLAT_ENABLE_REMOTE_SHUTDOWN", "0").lower() in ("1","true","yes")
SHUTDOWN_SECRET = os.environ.get("AXIPLAT_SHUTDOWN_SECRET", PASSCODE)

# Load existing tokens
if os.path.isfile(VERIFICATION_TOKEN_FILE):
    try:
        with open(VERIFICATION_TOKEN_FILE, 'r') as f:
            verification_tokens = json.load(f)
    except Exception:
        logger.exception("Could not read %s; starting with no tokens", VERIFICATION_TOKEN_FILE)
        verification_tokens = {}

# Helper routines
def adaptive_blob_mean(face_img):
    """Blend default mean subtraction with current frame stats."""
    if face_img is None or DYNAMIC_MEAN_BLEND <= 0:
        return DEFAULT_BLOB_MEAN
    b_mean, g_mean, r_mean, _ = cv2.mean(face_img)
    blended = []
    for default_val, current_val in zip(DEFAULT_BLOB_MEAN, (b_mean, g_mean, r_mean)):
        blended.append((1.0 - DYNAMIC_MEAN_BLEND) * default_val + DYNAMIC_MEAN_BLEND * current_val)
    return tuple(blended)


def soft_expected_age_from_dnn(face_img):
    """Raw model output: probability-weighted expected age from the DNN."""
    if age_net is None:
        return None
    try:
        mean_vals = adaptive_blob_mean(face_img)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227,227),
            mean_vals,
            swapRB=False
        )
        age_net.setInput(blob)
        preds = age_net.forward()[0]

        e = np.exp(preds - np.max(preds))
        probs = e / e.sum()

        # Always use the probability-weighted expected age: snapping to the
        # top bucket's midpoint made the running estimate jumpy, which
        # destabilized the EMA and the final decision.
        expected = float(np.dot(probs, AGE_MIDPOINTS))
        return expected
    except Exception:
        logger.exception("DNN age inference failed")
        return None

def cnn_regression_age(face_img):
    if age_cnn_model is None:
        return None
    try:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64,64))
        arr = img.astype("float32")/255.0
        arr = np.expand_dims(arr, axis=0)
        pred = age_cnn_model.predict(arr)
        return float(pred[0][0])
    except Exception:
        logger.exception("CNN age inference failed")
        return None

# ArcFace 5-point alignment template (insightface norm_crop target)
ARCFACE_DST = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])

def arcface_align(frame, landmarks):
    """112x112 similarity-aligned crop for genderage, from MediaPipe landmarks.

    Point order follows ArcFace: left eye, right eye, nose, left mouth,
    right mouth. Iris centers (refine_landmarks adds 468/473) are preferred.
    """
    h, w = frame.shape[:2]
    lm = landmarks.landmark
    try:
        if len(lm) >= 474:
            src = np.array([[lm[i].x * w, lm[i].y * h] for i in (468, 473, 1, 61, 291)],
                           dtype=np.float32)
        else:
            src = np.array([
                [(lm[33].x + lm[133].x) / 2 * w, (lm[33].y + lm[133].y) / 2 * h],
                [(lm[362].x + lm[263].x) / 2 * w, (lm[362].y + lm[263].y) / 2 * h],
                [lm[1].x * w, lm[1].y * h],
                [lm[61].x * w, lm[61].y * h],
                [lm[291].x * w, lm[291].y * h],
            ], dtype=np.float32)
        m, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST)
        if m is None:
            return None
        return cv2.warpAffine(frame, m, (112, 112), borderValue=0.0)
    except Exception:
        logger.exception("ArcFace alignment failed")
        return None

def genderage_age(frame, landmarks):
    """Raw age from insightface genderage.onnx, or None when unavailable."""
    if genderage_session is None:
        return None
    try:
        aligned = arcface_align(frame, landmarks)
        if aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0, GENDERAGE_INPUT_SIZE, (0.0, 0.0, 0.0), swapRB=True)
        out = genderage_session.run(None, {"data": blob})[0][0]
        return float(out[2]) * 100.0
    except Exception:
        logger.exception("genderage inference failed")
        return None

def crop_face_from_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]
    x_min, x_max = max(0,min(xs)), min(w-1,max(xs))
    y_min, y_max = max(0,min(ys)), min(h-1,max(ys))

    mw = int((x_max-x_min)*0.2)
    mh = int((y_max-y_min)*0.2)

    x1=max(0,x_min-mw); y1=max(0,y_min-mh)
    x2=min(w-1,x_max+mw); y2=min(h-1,y_max+mh)

    if x2-x1<10 or y2-y1<10:
        return None

    return frame[y1:y2, x1:x2].copy()

def normalize_face(face_img):
    try:
        lab=cv2.cvtColor(face_img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l2=clahe.apply(l)
        lab2=cv2.merge((l2,a,b))
        return cv2.cvtColor(lab2,cv2.COLOR_LAB2BGR)
    except Exception:
        logger.exception("Face normalization failed")
        return face_img

# MediaPipe FaceMesh landmark indices used by the quality gate
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263
LM_NOSE_TIP = 1

def assess_face_quality(frame, landmarks):
    """Check a detected face is usable for age estimation.

    Returns (face_crop, quality) where quality carries the size/pose checks.
    A face that is too far away or visibly turned produces unreliable age
    estimates, so those frames are excluded from sampling.
    """
    h, w, _ = frame.shape
    lm = landmarks.landmark
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    face_w = x_max - x_min

    size_ok = face_w > 0 and (face_w / w) >= MIN_FACE_WIDTH_RATIO

    left_eye_x = float(lm[LM_LEFT_EYE_OUTER].x * w)
    right_eye_x = float(lm[LM_RIGHT_EYE_OUTER].x * w)
    nose_x = float(lm[LM_NOSE_TIP].x * w)
    eye_dist = max(1.0, right_eye_x - left_eye_x)
    eye_mid_x = (left_eye_x + right_eye_x) / 2.0
    # Nose offset from the eye midpoint, scaled by eye distance: ~0 head-on,
    # grows as the head turns. Cheap yaw proxy that needs no solvePnP.
    yaw_ratio = (nose_x - eye_mid_x) / eye_dist
    pose_ok = abs(yaw_ratio) <= MAX_YAW_RATIO

    quality = {
        "size_ok": bool(size_ok),
        "pose_ok": bool(pose_ok),
        "yaw_ratio": float(yaw_ratio),
    }
    face = crop_face_from_landmarks(frame, landmarks)
    return face, quality

def reject_outliers_iqr(values):
    """Drop samples outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Keeps a single bad frame (sudden misread, partial blur) from dragging
    the median. Small sets are returned untouched.
    """
    if len(values) < 4:
        return list(values)
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi]

def ema_update(prev,val,alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev*(1-alpha)+val*alpha

def group_from_median(m):
    diffs=[abs(m-mp) for mp in AGE_MIDPOINTS]
    idx=int(np.argmin(diffs))
    return AGE_LIST[idx]

# Reference-point calibration helpers -----------------------------------------

def load_calibration_points(model_name):
    """Load (measured, true) reference points recorded for this estimator."""
    if not os.path.isfile(CALIBRATION_FILE):
        return []
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)
        return [
            (float(p["measured"]), float(p["true"]))
            for p in data.get("points", [])
            if p.get("model") == model_name
        ]
    except Exception:
        logger.exception("Could not read %s", CALIBRATION_FILE)
        return []

def fit_calibration(points):
    """Fit age' = scale * age + offset from (measured, true) points.

    One point (or no variance) -> offset-only; two distinct points ->
    least-squares. Scale is clamped to a sane range with a warning.
    """
    if not points:
        return 1.0, 0.0
    measured = np.asarray([p[0] for p in points], dtype=float)
    true = np.asarray([p[1] for p in points], dtype=float)
    if len(points) >= 2 and np.ptp(measured) > 1e-6 and np.ptp(true) > 1e-6:
        scale, offset = np.polyfit(measured, true, 1)
    else:
        # All references share one true age (or one measurement): a slope
        # fit would be degenerate, so shift-only is the honest fix.
        scale, offset = 1.0, float(np.mean(true) - np.mean(measured))
    if not (CALIBRATION_SCALE_MIN <= scale <= CALIBRATION_SCALE_MAX):
        logger.warning("Fitted calibration scale %.3f outside [%s, %s]; clamping",
                       scale, CALIBRATION_SCALE_MIN, CALIBRATION_SCALE_MAX)
        scale = float(np.clip(scale, CALIBRATION_SCALE_MIN, CALIBRATION_SCALE_MAX))
    return float(scale), float(offset)

def load_effective_calibration(model_name):
    """Effective (scale, offset): env vars > calibration.json points > identity."""
    env_scale = os.environ.get("AGE_CALIBRATION_SCALE")
    env_offset = os.environ.get("AGE_CALIBRATION_OFFSET")
    if env_scale is not None or env_offset is not None:
        scale = float(env_scale) if env_scale is not None else 1.0
        offset = float(env_offset) if env_offset is not None else 0.0
        logger.info("Calibration from env: scale=%.3f offset=%.2f", scale, offset)
        return scale, offset
    points = load_calibration_points(model_name)
    if points:
        scale, offset = fit_calibration(points)
        logger.info("Calibration from %s (%d %s point(s)): scale=%.3f offset=%.2f",
                    CALIBRATION_FILE, len(points), model_name, scale, offset)
        return scale, offset
    logger.info("Calibration: identity (no env override, no %s points for %s)",
                CALIBRATION_FILE, model_name)
    return 1.0, 0.0

def apply_calibration(age):
    return CALIBRATION_SCALE * age + CALIBRATION_OFFSET

