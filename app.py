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
