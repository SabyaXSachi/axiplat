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

