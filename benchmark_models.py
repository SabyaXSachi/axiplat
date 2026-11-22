"""Benchmark the old Levi-Hassner Caffe DNN vs insightface genderage.onnx.

Grabs frames from the webcam (with the same face-quality gating as app.py),
runs both estimators on the same faces, and prints raw outputs side by side.
For genderage the raw fc1 vector is dumped first so the age decoding can be
confirmed empirically before being trusted.

Run:  python benchmark_models.py [num_frames]
"""

import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

import app as axiplat  # reuses face_mesh, quality gate, raw Caffe estimate

GENDERAGE_PATH = "model/genderage.onnx"

# ArcFace 5-point alignment template (standard insightface norm_crop target)
ARCFACE_DST = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
])


def arcface_align(frame, landmarks):
    """112x112 similarity-aligned crop using MediaPipe landmarks.

    Point order follows ArcFace: left eye, right eye, nose, left mouth,
    right mouth. Iris centers (landmarks 468/473) are used when available.
    """
    h, w = frame.shape[:2]
    lm = landmarks.landmark
    try:
        if len(lm) >= 474:  # refine_landmarks=True adds iris points
            pts = [(468, ), (473, ), (1, ), (61, ), (291, )]
            src = np.array([[lm[i[0]].x * w, lm[i[0]].y * h] for i in pts],
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
        return None


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    ga_session = None
    ga_in_size = (96, 96)
    try:
        ga_session = ort.InferenceSession(
            GENDERAGE_PATH, providers=["CPUExecutionProvider"])
        shape = ga_session.get_inputs()[0].shape  # e.g. ['N', 3, 96, 96]
        ga_in_size = (shape[3], shape[2])  # (w, h)
        print(f"genderage.onnx loaded (input {ga_in_size[0]}x{ga_in_size[1]})")
    except Exception as e:
        print(f"genderage.onnx not available: {e}")

    cap = cv2.VideoCapture(axiplat.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera unavailable")
        return

    caffe_ages = []
    ga_rows = []  # raw fc1 vectors
    frame_h, frame_w = None, None

    print(f"\nCollecting {n_frames} quality-passing frames - look at the camera...")
    while len(caffe_ages) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = axiplat.face_mesh.process(rgb)
        if not (res and res.multi_face_landmarks):
            continue
        lm = res.multi_face_landmarks[0]
        face, q = axiplat.assess_face_quality(frame, lm)
        if face is None or not (q["size_ok"] and q["pose_ok"]):
            continue

        face = axiplat.normalize_face(face)
        caffe_age = axiplat.soft_expected_age_from_dnn(face)
        if caffe_age is None:
            continue

        caffe_ages.append(caffe_age)
        if ga_session is not None:
            aligned = arcface_align(frame, lm)
            if aligned is not None:
                blob = cv2.dnn.blobFromImage(
                    aligned, 1.0, ga_in_size, (0.0, 0.0, 0.0), swapRB=True)
                out = ga_session.run(None, {"data": blob})[0][0]
                ga_rows.append(out.tolist())

        if frame_w is None:
            frame_h, frame_w = frame.shape[:2]
        print(f"  frame {len(caffe_ages):2d}: caffe={caffe_age:5.1f}  "
              f"fc1={np.round(ga_rows[-1], 4).tolist() if ga_rows else 'n/a'}")
        time.sleep(0.25)

    cap.release()

    caffe_arr = np.asarray(caffe_ages)
    print("\n=== Levi-Hassner Caffe DNN (RAW, no calibration) ===")
    print(f"  n={len(caffe_arr)}  mean={caffe_arr.mean():.2f}  "
          f"median={np.median(caffe_arr):.2f}  std={caffe_arr.std():.2f}  "
          f"range=[{caffe_arr.min():.1f}, {caffe_arr.max():.1f}]")

    if ga_rows:
        ga = np.asarray(ga_rows)
        print("\n=== genderage.onnx raw fc1 columns ===")
        for c in range(ga.shape[1]):
            col = ga[:, c]
            print(f"  fc1[{c}]: mean={col.mean():.4f}  "
                  f"range=[{col.min():.4f}, {col.max():.4f}]")
        print("\nDecoding candidates (mean over frames):")
        print(f"  fc1[1] * 100          -> {ga[:, 1].mean() * 100:.1f}")
        print(f"  fc1[2] * 100          -> {ga[:, 2].mean() * 100:.1f}")
        print(f"  sum(fc1[1:]) * 100    -> {ga[:, 1:].sum(axis=1).mean() * 100:.1f}")
        print(f"  fc1[1] * 100 + fc1[2] -> {ga[:, 1].mean() * 100 + ga[:, 2].mean():.1f}")
        print(f"  sum(fc1)              -> {ga.sum(axis=1).mean():.1f}")
    else:
        print("\ngenderage.onnx produced no outputs")


if __name__ == "__main__":
    main()
