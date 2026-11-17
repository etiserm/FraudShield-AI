from __future__ import annotations

from typing import Any

import numpy as np


LEGACY_POLICY_ID = "f1_balanced"


def decision_policies(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return validated policy metadata, including a fallback for version 1 artifacts."""
    raw = bundle.get("decision_policies")
    if raw is None:
        raw = {
            LEGACY_POLICY_ID: {
                "label": "Balanced — maximize validation F1",
                "description": "The threshold was selected on validation data by maximizing F1.",
                "threshold": bundle.get("threshold"),
                "selection_method": "maximum validation F1",
            }
        }

    if not isinstance(raw, dict) or not raw:
        raise ValueError("The model bundle contains invalid decision policies.")

    policies: dict[str, dict[str, Any]] = {}
    for policy_id, policy in raw.items():
        if not isinstance(policy_id, str) or not policy_id or len(policy_id) > 64:
            raise ValueError("The model bundle contains an invalid decision policy ID.")
        if not isinstance(policy, dict):
            raise ValueError("The model bundle contains invalid decision policy details.")
        label = policy.get("label")
        description = policy.get("description", "")
        threshold = policy.get("threshold")
        if not isinstance(label, str) or not label or len(label) > 160:
            raise ValueError("The model bundle contains an invalid decision policy label.")
        if not isinstance(description, str) or len(description) > 500:
            raise ValueError("The model bundle contains an invalid decision policy description.")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("The model bundle contains an invalid policy threshold.") from error
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The model bundle contains an invalid policy threshold.")
        validated_policy = {**policy, "threshold": threshold}
        if "test_metrics" in policy:
            validated_policy["test_metrics"] = _validate_test_metrics(policy["test_metrics"])
        policies[policy_id] = validated_policy

    default_id = bundle.get("default_policy", LEGACY_POLICY_ID)
    if not isinstance(default_id, str) or default_id not in policies:
        raise ValueError("The model bundle default decision policy is unavailable.")
    return policies


def _validate_test_metrics(metrics: Any) -> dict[str, Any]:
    """Validate aggregate held-out metrics before exposing them to the local UI."""
    required_scores = ("precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy")
    if not isinstance(metrics, dict):
        raise ValueError("The model bundle contains invalid held-out test metrics.")

    clean: dict[str, Any] = {}
    for name in required_scores:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("The model bundle contains invalid held-out test metrics.") from error
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("The model bundle contains invalid held-out test metrics.")
        clean[name] = value

    counts = ("rows", "fraud_support")
    for name in counts:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("The model bundle contains invalid held-out test counts.")
        clean[name] = int(value)

    try:
        raw_matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The model bundle contains an invalid held-out confusion matrix.") from error
    if (
        raw_matrix.shape != (2, 2)
        or not np.isfinite(raw_matrix).all()
        or (raw_matrix < 0).any()
        or (raw_matrix > clean["rows"]).any()
        or not np.equal(raw_matrix, np.floor(raw_matrix)).all()
    ):
        raise ValueError("The model bundle contains an invalid held-out confusion matrix.")
    matrix = raw_matrix.astype(np.int64)
    if int(matrix.sum()) != clean["rows"] or int(matrix[1].sum()) != clean["fraud_support"]:
        raise ValueError("The held-out test counts do not match the confusion matrix.")

    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    expected_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    expected_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall else 0.0
    )
    expected_accuracy = (true_negative + true_positive) / clean["rows"]
    for name, expected in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
        ("accuracy", expected_accuracy),
    ):
        if not np.isclose(clean[name], expected, rtol=0, atol=1e-8):
            raise ValueError("Held-out test metrics do not match the confusion matrix.")

    clean["confusion_matrix"] = matrix.tolist()
    return clean


def selected_policy(
    bundle: dict[str, Any], policy_id: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Resolve a requested policy, defaulting to the bundle's declared policy."""
    policies = decision_policies(bundle)
    chosen_id = policy_id or bundle.get("default_policy", LEGACY_POLICY_ID)
    if not isinstance(chosen_id, str) or chosen_id not in policies:
        raise ValueError("Choose one of the available review policies.")
    return chosen_id, policies[chosen_id]
