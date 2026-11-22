"""
Age Estimation Model Visualization Script
Creates comprehensive matplotlib visualizations for the age gate application results.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2

# Try to import models (same as app.py)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Configuration (matching app.py)
MODEL_DIR = os.path.join(os.getcwd(), "model")
AGE_PROTO = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_CAFFE = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MIDPOINTS = [(int(a.split('(')[1].split('-')[0]) + int(a.split('-')[1].split(')')[0])) / 2.0 
                 for a in AGE_LIST]

DNN_WEIGHT = 0.6
CNN_WEIGHT = 0.4
CALIBRATION_SCALE = 1.0
CALIBRATION_OFFSET = 0.0
EMA_ALPHA = 0.3
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = 20
MIN_AGE_THRESHOLD = 14.0
SESSION_LOG_FILE = os.path.join(os.getcwd(), "session_log.json")

# Model loading
age_net = None
USE_DNN = False
if os.path.isfile(AGE_PROTO) and os.path.isfile(AGE_CAFFE):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_CAFFE)
        USE_DNN = True
        print("✓ Loaded OpenCV DNN age model.")
    except Exception as e:
        print(f"✗ Failed to load DNN age model: {e}")

age_cnn_model = None
if TF_AVAILABLE:
    cnn_path = os.path.join(MODEL_DIR, "cnn_age_model.h5")
    if os.path.isfile(cnn_path):
        try:
            age_cnn_model = load_model(cnn_path)
            print("✓ Loaded CNN regression model.")
        except Exception as e:
            print(f"✗ Failed to load CNN: {e}")
    else:
        print("ℹ No CNN model found.")
else:
    print("ℹ TensorFlow not available; CNN regression disabled.")

# Helper functions (simplified from app.py)
def adaptive_blob_mean(face_img):
    DEFAULT_BLOB_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
    DYNAMIC_MEAN_BLEND = 0.35
    if face_img is None or DYNAMIC_MEAN_BLEND <= 0:
        return DEFAULT_BLOB_MEAN
    b_mean, g_mean, r_mean, _ = cv2.mean(face_img)
    blended = []
    for default_val, current_val in zip(DEFAULT_BLOB_MEAN, (b_mean, g_mean, r_mean)):
        blended.append((1.0 - DYNAMIC_MEAN_BLEND) * default_val + DYNAMIC_MEAN_BLEND * current_val)
    return tuple(blended)

def brightness_age_adjust(face_img):
    BRIGHTNESS_REFERENCE = 132.0
    BRIGHTNESS_CORRECTION = 0.018
    BRIGHTNESS_CORRECTION_CLAMP = 6.0
    if face_img is None or BRIGHTNESS_CORRECTION == 0:
        return 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    delta = BRIGHTNESS_REFERENCE - float(np.mean(gray))
    adjustment = delta * BRIGHTNESS_CORRECTION
    return float(np.clip(adjustment, -BRIGHTNESS_CORRECTION_CLAMP, BRIGHTNESS_CORRECTION_CLAMP))

def get_dnn_predictions(face_img):
    """Get DNN model predictions with probabilities"""
    if age_net is None:
        return None, None, None
    try:
        mean_vals = adaptive_blob_mean(face_img)
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), mean_vals, swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()[0]

        e = np.exp(preds - np.max(preds))
        probs = e / e.sum()

        expected = float(np.dot(probs, AGE_MIDPOINTS))
        calibrated = CALIBRATION_SCALE * expected + CALIBRATION_OFFSET
        calibrated += brightness_age_adjust(face_img)

        return float(calibrated), probs, expected
    except Exception:
        return None, None, None

def get_cnn_prediction(face_img):
    """Get CNN model prediction"""
    if age_cnn_model is None:
        return None
    try:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        arr = img.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = age_cnn_model.predict(arr, verbose=0)
        return float(pred[0][0])
    except Exception:
        return None

def ema_update(prev, val, alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev * (1 - alpha) + val * alpha

def simulate_age_predictions(num_samples=20, base_age=25, noise_std=2.0):
    """Simulate age predictions over time for demonstration"""
    times = np.arange(0, num_samples * SAMPLE_INTERVAL, SAMPLE_INTERVAL)
    
    # Simulate DNN and CNN predictions with some variation
    dnn_ages = base_age + np.random.normal(0, noise_std, num_samples)
    cnn_ages = base_age + np.random.normal(0, noise_std * 1.2, num_samples)
    
    # Combined weighted average
    combined_ages = DNN_WEIGHT * dnn_ages + CNN_WEIGHT * cnn_ages
    
    # Calculate EMA
    ema_ages = []
    ema_val = None
    for age in combined_ages:
        ema_val = ema_update(ema_val, age)
        ema_ages.append(ema_val)
    
    # Simulate DNN probabilities
    dnn_probs_list = []
    for i in range(num_samples):
        # Create realistic probability distribution
        probs = np.random.dirichlet([1, 1, 1, 2, 3, 2, 1, 1])  # Favor middle ages
        dnn_probs_list.append(probs)
    
    return {
        'times': times,
        'dnn_ages': dnn_ages,
        'cnn_ages': cnn_ages,
        'combined_ages': combined_ages,
        'ema_ages': np.array(ema_ages),
        'dnn_probs': np.array(dnn_probs_list)
    }

def load_real_session(path=SESSION_LOG_FILE):
    """Load a real verification session recorded by app.py (session_log.json).

    Returns (data, meta) where data matches the simulate_age_predictions()
    shape and meta carries the session's final decision, or (None, None)
    when no session log exists.
    """
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, 'r') as f:
            log = json.load(f)
    except Exception as e:
        print(f"✗ Could not read {path}: {e}")
        return None, None

    samples = log.get("samples") or []
    combined = [s["age"] for s in samples if s.get("age") is not None]
    if not combined:
        print(f"ℹ {path} contains no usable samples.")
        return None, None

    times = [s.get("t", i * SAMPLE_INTERVAL) for i, s in enumerate(samples)
             if s.get("age") is not None]
    ga = [s["ga"] for s in samples if s.get("ga") is not None]
    dnn = [s["dnn"] for s in samples if s.get("dnn") is not None]
    cnn = [s["cnn"] for s in samples if s.get("cnn") is not None]
    # The active model's curve gets the headline: genderage when present,
    # otherwise the Caffe fallback that produced the measurement.
    if ga:
        primary, primary_label = ga, "genderage (primary)"
    elif dnn:
        primary, primary_label = dnn, "caffe-dnn (fallback)"
    else:
        primary, primary_label = combined, "fused (calibrated)"

    ema_ages = []
    ema_val = None
    for age in combined:
        ema_val = ema_update(ema_val, age)
        ema_ages.append(ema_val)

    data = {
        'times': np.array(times, dtype=float),
        'dnn_ages': np.array(primary, dtype=float),
        'primary_label': primary_label,
        'cnn_ages': np.array(cnn, dtype=float) if cnn else np.array(combined, dtype=float),
        'combined_ages': np.array(combined, dtype=float),
        'ema_ages': np.array(ema_ages, dtype=float),
        'dnn_probs': np.array([]),  # per-sample probabilities are not logged
    }
    meta = {
        'source': 'real',
        'session_id': log.get("session_id"),
        'finished_at': log.get("finished_at"),
        'final_median_age': log.get("final_median_age"),
        'access_allowed': log.get("access_allowed"),
        'threshold': log.get("threshold", MIN_AGE_THRESHOLD),
        'valid_count': log.get("valid_count"),
        'outlier_filtered_count': log.get("outlier_filtered_count"),
        'camera_error': log.get("camera_error"),
    }
    print(f"✓ Loaded real session log: {path} ({len(combined)} samples)")
    return data, meta

