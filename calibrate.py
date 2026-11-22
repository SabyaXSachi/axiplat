"""Manage reference-age calibration for AXIPLAT.

Usage:
    python calibrate.py 21      # add a point: true age 21, measured value
                                # taken from the latest session_log.json
    python calibrate.py list    # show points + current fit + mapping examples
    python calibrate.py clear   # remove all points (back to identity)

Points are stamped with the estimator that produced the measurement
(session_log.json's primary_model), so switching estimators automatically
ignores stale points instead of mis-calibrating.
"""

import json
import os
import sys

CALIBRATION_FILE = "calibration.json"
SESSION_LOG_FILE = "session_log.json"
SCALE_MIN, SCALE_MAX = 0.3, 2.0


def fit_calibration(points):
    """Mirror of app.fit_calibration: (measured, true) -> (scale, offset)."""
    import numpy as np

    if not points:
        return 1.0, 0.0
    measured = np.asarray([p[0] for p in points], dtype=float)
    true = np.asarray([p[1] for p in points], dtype=float)
    if len(points) >= 2 and np.ptp(measured) > 1e-6 and np.ptp(true) > 1e-6:
        scale, offset = np.polyfit(measured, true, 1)
    else:
        scale, offset = 1.0, float(np.mean(true) - np.mean(measured))
    scale = float(np.clip(scale, SCALE_MIN, SCALE_MAX))
    return float(scale), float(offset)


def load_points(model_filter=None):
    if not os.path.isfile(CALIBRATION_FILE):
        return []
    with open(CALIBRATION_FILE, "r") as f:
        data = json.load(f)
    pts = [
        (float(p["measured"]), float(p["true"]), p.get("model", "?"))
        for p in data.get("points", [])
    ]
    if model_filter:
        pts = [p for p in pts if p[2] == model_filter]
    return pts


def save_points(points):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"points": points}, f, indent=2)


def show(points):
    # The app fits per-estimator (points must match the active model), so
    # show one fit per model instead of mixing them together.
    models = sorted({p[2] for p in points})
    for model in models:
        group = [(m, t) for m, t, pm in points if pm == model]
        scale, offset = fit_calibration(group)
        print(f"[{model}] {len(group)} point(s): scale={scale:.4f} offset={offset:.2f}")
        for m, t, pm in points:
            if pm == model:
                print(f"    measured {m:6.2f} -> true {t:5.1f}")
        for example in (10, 14, 21, 30, 45):
            print(f"    raw {example:3d} -> calibrated {scale * example + offset:6.1f}")
    if not points:
        print("No calibration points (identity).")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "clear":
        save_points([])
        print("Calibration points cleared (back to identity).")
        return

    if cmd == "list":
        show(load_points())
        return

    try:
        true_age = float(cmd)
    except ValueError:
        print(f"Unknown argument: {cmd!r}. Use a number, 'list', or 'clear'.")
        sys.exit(1)

    if not os.path.isfile(SESSION_LOG_FILE):
        print(f"No {SESSION_LOG_FILE} found - run a verification session first.")
        sys.exit(1)
    with open(SESSION_LOG_FILE, "r") as f:
        log = json.load(f)

    # Reference points must use RAW model outputs: final_median_age in the
    # log is already calibrated, and recording a calibrated value would
    # double-correct on the next fit. Prefer the raw primary-model samples.
    model = log.get("primary_model", "caffe-dnn")
    raw_field = "ga" if model == "genderage-onnx" else "dnn"
    raw_vals = [s[raw_field] for s in (log.get("samples") or [])
                if s.get(raw_field) is not None]
    if raw_vals:
        import numpy as np
        measured = float(np.median(raw_vals))
        source = f"raw {raw_field} median of {len(raw_vals)} samples"
    else:
        # Legacy logs that predate raw-field logging
        measured = log.get("final_median_age")
        source = "legacy final median"
    if measured is None:
        print("Latest session recorded no usable samples (no decision was made).")
        sys.exit(1)

    points = load_points()
    points.append((round(float(measured), 2), true_age, model))
    save_points([{"measured": m, "true": t, "model": pm} for m, t, pm in points])
    print(f"Added point: measured {measured:.2f} -> true {true_age} [{model}] ({source})")
    show(points)
    print("\nRestart the server for the new calibration to take effect.")


if __name__ == "__main__":
    main()
