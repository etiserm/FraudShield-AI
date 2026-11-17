# FraudShield AI — local HTML demo app

This is a local HTML/CSS/JavaScript website backed by a small Python API. The API loads the Random Forest model trained in the FraudShield Colab notebook and scores public or synthetic transaction rows; it does not train a model.

## Files

- `site/` — the HTML, CSS, and JavaScript interface.
- `api.py` — local API, CSV validation, model scoring, and result download.
- `desktop_app.py` — starts the local API inside a native macOS window.
- `assets/fraudshield_icon.svg` — editable vector source for the app icon.
- `assets/fraudshield_icon.icns` — macOS app icon included in the build.
- `build_macos_app.command` — builds `dist/FraudShield AI.app` for this Mac.
- `requirements-macos.txt` — dependencies for the standalone Mac app build.
- `app.py` — the earlier Streamlit prototype, kept as an alternate interface.
- `requirements.txt` — app dependencies. The scikit-learn version matches the Colab model.
- `fraudshield_rf_model.joblib` — created by the notebook export cell; keep it beside `app.py`.

## Run locally on macOS

Use Python 3.13 to match the Colab environment used for the model artifact. Double-click `start_app.command`, then open `http://127.0.0.1:8080` in your browser. Keep the Terminal window open while using the app; press `Ctrl+C` there to stop it.

```bash
cd fraudshield_web_app
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8080
```

The website stays on this Mac. Stop it with `Ctrl+C` in Terminal.

## Build the Mac app

Double-click `build_macos_app.command`. The first build installs the required Python packages and may take several minutes. It creates both `dist/FraudShield AI.app` and `dist/FraudShield AI-macOS-arm64.zip`. Unzip the archive and open `FraudShield AI.app`; do not open `site/index.html` directly. The app opens the same HTML interface in its own window, starts the model service on a private loopback port, and stops that service when the app closes. The build is specific to this Mac's processor and macOS version. It is ad-hoc signed, not signed with an Apple Developer ID or notarized, and is not prepared for App Store or public distribution. If macOS blocks the first launch, Control-click the app and choose Open.

## Input CSV

The file must contain exactly the model features `Time`, `V1`–`V28`, and `Amount`. An optional `Class` column is explicitly discarded before scoring. Files with missing, duplicate, unexpected, non-numeric, or non-finite feature values are rejected. Each upload is limited to 180 MiB and 300,000 rows.

The app uses the validation-selected Random Forest threshold from the notebook. Its scores are not calibrated probabilities, and flagged rows require human review. This is an educational prototype, not financial advice or a production fraud service.

These limits accommodate the complete public Kaggle CSV used by this project (about 151 MB and 284,807 rows). Larger files should be split into smaller public or synthetic batches.

## Model artifact safety

The app loads a fixed local `joblib` file. `joblib` model files can execute code when loaded, so only use the artifact exported by the trusted FraudShield notebook. The app does not accept model files from users. Do not upload real cardholder data, credentials, or private transactions.

## Public deployment

This starter is configured for local use only. Do not publish the app, model file, or repository without first reviewing repository visibility, model licensing, and data-use terms.
