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

