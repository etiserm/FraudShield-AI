from __future__ import annotations

import base64
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from model_utils import decision_policies, selected_policy


APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
STATIC_DIR = APP_DIR / "site"
MODEL_PATH = APP_DIR / "fraudshield_rf_model.joblib"
MAX_UPLOAD_BYTES = 180 * 1024 * 1024
MAX_ROWS = 300_000
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

app = FastAPI(title="FraudShield AI", docs_url=None, redoc_url=None)
logger = logging.getLogger("fraudshield.api")


def load_trusted_bundle() -> dict[str, Any]:
    """Load and validate the model artifact exported by the project notebook."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The exported FraudShield model file is missing.")

    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "feature_columns", "threshold", "schema_version"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("The model file is not a supported FraudShield model bundle.")
    if bundle["schema_version"] not in {1, 2}:
        raise ValueError("The model bundle version is not supported by this app.")
    if bundle["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("The model feature order does not match the website's CSV schema.")

    threshold = float(bundle["threshold"])
    model = bundle["model"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The model bundle contains an invalid decision threshold.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("The model bundle does not contain a probability-scoring classifier.")
    if 1 not in model.classes_ or int(model.n_features_in_) != len(FEATURE_COLUMNS):
        raise ValueError("The model bundle has an invalid fraud class or feature count.")
    if bundle["schema_version"] == 2 and "decision_policies" not in bundle:
        raise ValueError("The model bundle is missing its decision policies.")
    decision_policies(bundle)
    return bundle


try:
    MODEL_BUNDLE: dict[str, Any] | None = load_trusted_bundle()
    MODEL_ERROR: str | None = None
except Exception as error:
    logger.exception("The local FraudShield model bundle could not be loaded.")
    MODEL_BUNDLE = None
    # Keep local file paths and deserialization details out of browser responses.
    MODEL_ERROR = "The local model could not be loaded. Restart the app or replace its model file."


def validate_csv(file_bytes: bytes) -> tuple[pd.DataFrame, bool]:
    """Validate CSV input and return only the model's feature columns."""
    if not file_bytes:
        raise ValueError("The uploaded CSV is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV exceeds the 180 MiB upload limit. Upload a smaller batch.")

    try:
        raw_df = pd.read_csv(BytesIO(file_bytes))
    except Exception as error:
        raise ValueError(f"Could not read the CSV file: {error}") from error

    if raw_df.empty:
        raise ValueError("The CSV has a header but no transaction rows.")
    if not raw_df.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names.")
    if len(raw_df) > MAX_ROWS:
        raise ValueError(f"The website accepts up to {MAX_ROWS:,} rows per upload.")

    label_ignored = "Class" in raw_df.columns
    if label_ignored:
        raw_df = raw_df.drop(columns=["Class"])

    missing = [column for column in FEATURE_COLUMNS if column not in raw_df.columns]
    unexpected = [column for column in raw_df.columns if column not in FEATURE_COLUMNS]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError("CSV schema does not match the model (" + "; ".join(details) + ").")

    features = raw_df.loc[:, FEATURE_COLUMNS].copy()
    try:
        for column in FEATURE_COLUMNS:
            features[column] = pd.to_numeric(features[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All required feature values must be numeric.") from error

    if features.isna().any().any():
        raise ValueError("The CSV contains missing feature values.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("The CSV contains infinite or non-finite feature values.")
    for column in ("Time", "Amount"):
        if (features[column] < 0).any():
            raise ValueError(f"{column} values must be non-negative.")
    return features, label_ignored


def predict(features: pd.DataFrame, policy_id: str | None = None) -> pd.DataFrame:
    """Score validated rows with a threshold selected on validation data."""
    if MODEL_BUNDLE is None:
        raise RuntimeError(MODEL_ERROR or "The model could not be loaded.")

    model = MODEL_BUNDLE["model"]
    policy_id, policy = selected_policy(MODEL_BUNDLE, policy_id)
    threshold = float(policy["threshold"])
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(features.loc[:, FEATURE_COLUMNS])[:, positive_index]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The model returned an invalid score. Check the model artifact.")

    decisions = (scores >= threshold).astype(int)
    return pd.DataFrame(
        {
            "Transaction row": np.arange(1, len(features) + 1),
            "Fraud score (uncalibrated)": scores,
            "Decision policy": policy["label"],
            "Decision threshold": threshold,
            "Predicted class": decisions,
            "Predicted label": np.where(decisions == 1, "Fraud", "Normal"),
        }
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    if MODEL_BUNDLE is None:
        return {"status": "model_unavailable", "message": MODEL_ERROR}
    policies = decision_policies(MODEL_BUNDLE)
    default_id, default_policy = selected_policy(MODEL_BUNDLE)
    return {
        "status": "ready",
        "model": MODEL_BUNDLE.get("model_name", "Random Forest"),
        "default_policy": default_id,
        "threshold": float(default_policy["threshold"]),
        "decision_policies": [
            {
                "id": policy_id,
                "label": policy["label"],
                "description": policy.get("description", ""),
                "threshold": float(policy["threshold"]),
                **({"test_metrics": policy["test_metrics"]} if "test_metrics" in policy else {}),
            }
            for policy_id, policy in policies.items()
        ],
    }


@app.post("/api/predict")
def predict_csv(
    file: UploadFile = File(...),
    decision_policy: str | None = Form(default=None),
) -> Response:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "The model could not be loaded.")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload one CSV file.")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The CSV exceeds the 180 MiB upload limit.")

    try:
        file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
        features, label_ignored = validate_csv(file_bytes)
        predictions = predict(features, decision_policy)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("FraudShield prediction failed unexpectedly.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed unexpectedly. Check the model and input, then try again.",
        ) from error

    flagged = int((predictions["Predicted class"] == 1).sum())
    policy_id, policy = selected_policy(MODEL_BUNDLE, decision_policy)
    preview = []
    for record in predictions.head(10).to_dict(orient="records"):
        preview.append(
            {
                "transaction_row": int(record["Transaction row"]),
                "fraud_score": float(record["Fraud score (uncalibrated)"]),
                "threshold": float(record["Decision threshold"]),
                "predicted_label": str(record["Predicted label"]),
            }
        )
    preview_header = base64.urlsafe_b64encode(
        json.dumps(preview, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    csv_bytes = predictions.to_csv(index=False).encode("utf-8")

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="fraudshield_predictions.csv"',
            "X-Total-Transactions": str(len(predictions)),
            "X-Flagged-Count": str(flagged),
            "X-Flag-Rate": f"{flagged / len(predictions):.6f}",
            "X-Decision-Threshold": f"{float(policy['threshold']):.6f}",
            "X-Decision-Policy": policy_id,
            "X-Class-Ignored": str(label_ignored).lower(),
            "X-Preview-Base64": preview_header,
        },
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="site")
