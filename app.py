from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import decision_policies, selected_policy


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "fraudshield_rf_model.joblib"
MAX_UPLOAD_BYTES = 180 * 1024 * 1024
MAX_ROWS = 300_000
OPTIONAL_LABEL_COLUMN = "Class"


st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model_bundle(model_path: str, modified_at: float) -> dict[str, Any]:
    """Load the trusted model artifact exported by the project notebook."""
    del modified_at  # Included in the cache key so replacing the file reloads it.
    bundle = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "threshold", "schema_version"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("The model file is not a supported FraudShield model bundle.")
    if bundle["schema_version"] not in {1, 2}:
        raise ValueError("The model bundle version is not supported by this app.")
    if bundle["schema_version"] == 2 and "decision_policies" not in bundle:
        raise ValueError("The model bundle is missing its decision policies.")

    feature_columns = bundle["feature_columns"]
    expected_features = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or len(feature_columns) != len(set(feature_columns))
        or OPTIONAL_LABEL_COLUMN in feature_columns
        or feature_columns != expected_features
    ):
        raise ValueError("The model bundle contains an invalid feature list.")

    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The model bundle contains an invalid decision threshold.")

    model = bundle["model"]
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("The model bundle does not contain a probability-scoring classifier.")
    if 1 not in model.classes_:
        raise ValueError("The model bundle has no positive fraud class (Class = 1).")
    if int(model.n_features_in_) != len(feature_columns):
        raise ValueError("The model feature count does not match its feature list.")
    decision_policies(bundle)

    return bundle


def validate_uploaded_csv(
    file_bytes: bytes,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, bool]:
    """Validate a transaction CSV and return only model input features."""
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
        raise ValueError(f"The app accepts up to {MAX_ROWS:,} rows per upload.")

    includes_label = OPTIONAL_LABEL_COLUMN in raw_df.columns
    if includes_label:
        # The true label must never be passed to the prediction model.
        raw_df = raw_df.drop(columns=[OPTIONAL_LABEL_COLUMN])

    missing_columns = [column for column in feature_columns if column not in raw_df.columns]
    unexpected_columns = [column for column in raw_df.columns if column not in feature_columns]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if unexpected_columns:
            details.append(f"unexpected columns: {unexpected_columns}")
        raise ValueError("CSV schema does not match this model (" + "; ".join(details) + ").")

    features = raw_df.loc[:, feature_columns].copy()
    try:
        for column in feature_columns:
            features[column] = pd.to_numeric(features[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("All required feature values must be numeric.") from error

    if features.isna().any().any():
        raise ValueError("The CSV contains missing feature values.")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("The CSV contains infinite or non-finite feature values.")
    for nonnegative_column in ("Time", "Amount"):
        if nonnegative_column in features and (features[nonnegative_column] < 0).any():
            raise ValueError(f"{nonnegative_column} values must be non-negative.")

    return features, includes_label


def score_transactions(
    bundle: dict[str, Any], features: pd.DataFrame, policy_id: str | None = None
) -> pd.DataFrame:
    """Return scores and decisions using a validation-selected policy."""
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    _, policy = selected_policy(bundle, policy_id)
    threshold = float(policy["threshold"])
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(features.loc[:, feature_columns])[:, positive_index]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("The model returned an invalid score. Check the model artifact.")

    predicted_classes = (scores >= threshold).astype(int)
    return pd.DataFrame(
        {
            "Transaction row": np.arange(1, len(features) + 1),
            "Fraud score (uncalibrated)": scores,
            "Decision policy": policy["label"],
            "Decision threshold": threshold,
            "Predicted class": predicted_classes,
            "Predicted label": np.where(predicted_classes == 1, "Fraud", "Normal"),
        }
    )


st.title("🛡️ FraudShield AI")
st.subheader("Credit card fraud detection prototype")
st.info(
    "Educational prototype only — not financial advice and not for real payment decisions. "
    "Use public or synthetic data only. Model scores are not calibrated probabilities."
)

if not MODEL_PATH.is_file():
    st.error("The exported model file is missing.")
    st.markdown(
        "Export `fraudshield_rf_model.joblib` from the FraudShield Colab notebook, "
        "then place it next to `app.py`."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load the FraudShield model: {error}")
    st.stop()

policies = decision_policies(bundle)
default_policy_id = bundle.get("default_policy", "f1_balanced")
policy_ids = list(policies)
default_policy_index = policy_ids.index(default_policy_id)
selected_policy_id = st.selectbox(
    "Review policy",
    options=policy_ids,
    index=default_policy_index,
    format_func=lambda policy_id: policies[policy_id]["label"],
)
policy = policies[selected_policy_id]
threshold = float(policy["threshold"])
st.caption(
    f"Model: {bundle.get('model_name', 'Random Forest')} · "
    f"Validation-selected threshold: {threshold:.6f} · "
    f"Maximum upload: {MAX_ROWS:,} rows / 180 MiB"
)
st.caption(policy.get("description", ""))
if selected_policy_id == "cost_10_to_1":
    st.info(
        "The 10:1 ratio is illustrative, not a measured estimate of bank losses. "
        "On the held-out test period it detected the same 39 of 52 fraud cases as the balanced policy, "
        "but produced 16 false alerts versus 5."
    )

uploaded_file = st.file_uploader(
    "Upload a transaction CSV",
    type=["csv"],
    accept_multiple_files=False,
    help=(
        "Include the model's feature columns: Time, V1–V28, and Amount. "
        "The optional Class column is ignored and never used for prediction. "
        "The full public Kaggle dataset fits within this app's row and file-size limits."
    ),
)

if uploaded_file is None:
    st.markdown(
        "Upload a CSV containing `Time`, `V1`–`V28`, and `Amount`. "
        "The optional `Class` label is discarded before scoring."
    )
else:
    if st.button("Score transactions", type="primary"):
        try:
            if uploaded_file.size is not None and uploaded_file.size > MAX_UPLOAD_BYTES:
                raise ValueError("The CSV exceeds the 180 MiB upload limit. Upload a smaller batch.")
            features, label_was_ignored = validate_uploaded_csv(
                uploaded_file.getvalue(), bundle["feature_columns"]
            )
            predictions = score_transactions(bundle, features, selected_policy_id)
        except Exception as error:
            st.error(str(error))
            st.stop()

        if label_was_ignored:
            st.warning("The uploaded `Class` column was ignored; predictions used features only.")

        flagged = int((predictions["Predicted class"] == 1).sum())
        total = len(predictions)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Transactions scored", f"{total:,}")
        col_b.metric("Flagged for review", f"{flagged:,}")
        col_c.metric("Flag rate", f"{flagged / total:.2%}")

        st.caption(
            "A flagged row is a model alert for human review, not a confirmed fraud finding."
        )
        st.dataframe(predictions.head(500), use_container_width=True, hide_index=True)
        if total > 500:
            st.caption("Showing the first 500 rows. Download the full results below.")
        st.download_button(
            "Download predictions as CSV",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_predictions.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "FraudShield AI · Public Kaggle dataset · Human review required · "
    "Do not upload real cardholder data, credentials, or private transactions."
)
