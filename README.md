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

## Local Setup

```bash
git clone https://github.com/Hari-Narayana-Rath/axiplat.git
cd axiplat
```

Create a virtual environment (replace `python` with `python3` if needed):

```bash
python -m venv venv
```

Activate it:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies (Python 3.12 works; MediaPipe is pinned to a 3.12-compatible build):

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

or use a launcher: `start_server.bat` (visible), `start_server_hidden.bat` (background), `start_server_portable.bat` (auto-setup), `start_server.sh` (macOS/Linux). The control panel's Download Center links serve these files.

Flask runs on `http://127.0.0.1:5000/`. Open that URL in a browser, allow camera access, and follow the on-screen instructions. Press `Ctrl+C` in the terminal to stop the server. Debug mode is **off by default** (the Werkzeug debugger is a LAN security risk) — opt in with `FLASK_DEBUG=1`.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `AGE_MIN_THRESHOLD` | `14.0` | Minimum estimated age to grant access |
| `AGE_MIN_SAMPLES` | `6` | Good samples required before deciding |
| `AGE_MAX_SECONDS` | `20` | Camera session timeout |
| `AGE_EMA_ALPHA` | `0.3` | Smoothing factor for the live estimate |
| `AGE_MIN_FACE_RATIO` | `0.15` | Face width / frame width floor |
| `AGE_MAX_YAW_RATIO` | `0.25` | Head-turn rejection threshold |
| `AGE_TOKEN_HOURS` | `24` | Verification token validity |
| `AGE_CALIBRATION_SCALE` / `AGE_CALIBRATION_OFFSET` | from `calibration.json` | Override the reference-point fit (see calibration section) |
| `AGE_GATE_PASSCODE` | `admin123` | Parent override passcode — **change this** |
| `FLASK_DEBUG` | `0` | Enable Flask debug mode |

## Tests

```bash
python test_pipeline.py
```

Covers outlier rejection, face-quality gating, EMA behavior, calibration fitting and per-model guards, `/status` fields, and session-bound token issuing.

## Model comparison

```bash
python benchmark_models.py 20
```

Runs both estimators side by side on your webcam and prints raw outputs — use it to validate a model switch or a new calibration point.

## Real result plots

`python visualize_results.py` plots the **last real verification session** from `session_log.json` (written automatically by the server). Without a session log it falls back to clearly-labeled simulated data.

## Training the CNN (optional)

`train_cnn.py` **refuses to train without a real dataset** — a model trained on fake data would silently corrupt the age fusion. Download the [UTKFace aligned dataset](https://susanqq.github.io/UTKFace/), unzip it into `dataset/`, install TensorFlow, then run `python train_cnn.py`. It reports honest validation MAE and saves `model/cnn_age_model.h5`, which the app then blends in at 40% weight.

## Browser Extension (Optional)

The project includes a Chrome/Edge/Brave browser extension that intercepts **any Instagram page** (posts, reels, profiles — not just the homepage) and redirects to AXIPLAT if the user hasn't been verified.

### Setup Extension

1. Navigate to the `extension` folder
2. See `extension/README.md` for installation instructions
3. **Note:** Icon files are optional - the extension works without them

### How It Works

- When a user tries to visit Instagram, the extension checks for a valid verification token
- If not verified, they're redirected to AXIPLAT (`localhost:5000`)
- After successful age verification, a token is stored (valid for 24 hours)
- Instagram access is only allowed with a valid token
- Tokens are bound to the verification session that produced them — opening `/insta` without completing your own camera check never yields a token
