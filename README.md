# AXIPLAT

AXIPLAT is a Flask-based camera age verification stack that fuses OpenCV DNN predictions with an optional TensorFlow CNN regressor, plus a dark UI flow for redirecting to Instagram when access is granted.

## How verification works

1. Open the app, allow camera access — the server samples your face every 0.5 s for up to 20 s.
2. Only **quality-passing frames** are used: the face must be close enough (≥ 15% of frame width) and head-on (nose offset check rejects turned heads). Missed/turned frames are skipped entirely — they no longer drag the estimate down.
3. Age comes from **insightface's genderage.onnx** (buffalo_l pack) on an ArcFace-aligned face crop — on a known-age face its bias is ~5× smaller than the old Levi-Hassner Caffe DNN (now fallback). `benchmark_models.py` compares both on your webcam.
4. After at least **6 good samples** (and a stable 3-second window), outliers are dropped (IQR rule) and the **median** decides.
5. Estimated age ≥ **14** (configurable) grants access; a token is issued **only to that browser session** and stays valid 24 h.

**Honest limits**: enforcement ultimately lives in the browser extension — a determined kid can disable it, so combine AXIPLAT with OS-level parental controls for real resilience.

## Age calibration

Generic models carry systematic bias (e.g. young adults read +8 years on the old Caffe model). AXIPLAT corrects this with **reference points**: known true ages paired with what the active model measured.

```bash
python calibrate.py 21    # record: "the latest session measured me, I'm 21"
python calibrate.py list  # show points and the fitted mapping per model
python calibrate.py clear # reset to identity
```

Points live in `calibration.json` and are tagged with the estimator that produced them — switching models automatically ignores stale points (just re-run `calibrate.py <age>` once). Environment variables `AGE_CALIBRATION_SCALE` / `AGE_CALIBRATION_OFFSET` override the file.

