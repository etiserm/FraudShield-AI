const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#csv-file");
const dropzone = document.querySelector("#dropzone");
const fileTitle = document.querySelector("#file-title");
const fileHint = document.querySelector("#file-hint");
const analyzeButton = document.querySelector("#analyze-button");
const buttonLabel = document.querySelector(".button-label");
const errorPanel = document.querySelector("#error-panel");
const resultsPanel = document.querySelector("#results");
const downloadButton = document.querySelector("#download-button");
const previewBody = document.querySelector("#preview-body");
const policySelector = document.querySelector("#decision-policy");
const policyDescription = document.querySelector("#policy-description");
const thresholdDisplay = document.querySelector("#model-threshold");
const modelName = document.querySelector("#model-name");
const resultsNote = document.querySelector("#results-note");
const benchmarkPolicyLabel = document.querySelector("#benchmark-policy-label");
const benchmarkSummary = document.querySelector("#benchmark-summary");
const modelStatus = document.querySelector(".model-status");

const maxUploadBytes = 180 * 1024 * 1024;
let predictionBlob = null;
let availablePolicies = new Map();
let modelReady = false;
let isBusy = false;

function updatePolicyDisplay() {
  const policy = availablePolicies.get(policySelector.value);
  if (!policy) return;
  thresholdDisplay.textContent = Number(policy.threshold).toFixed(6);
  policyDescription.textContent = policy.description;
  resultsNote.textContent = `${policy.label}. A “Fraud” label is a model alert for human review, not a confirmed finding. Scores are not calibrated probabilities.`;

  const metrics = policy.test_metrics;
  const metricElements = [
    ["precision", document.querySelector("#benchmark-precision")],
    ["recall", document.querySelector("#benchmark-recall")],
    ["f1", document.querySelector("#benchmark-f1")],
    ["pr_auc", document.querySelector("#benchmark-pr-auc")],
    ["roc_auc", document.querySelector("#benchmark-roc-auc")],
  ];
  const policyName = policySelector.value === "cost_10_to_1" ? "COST SCENARIO" : "BALANCED POLICY";
  benchmarkPolicyLabel.textContent = `${policyName} · HELD-OUT TEST PERIOD`;
  if (!metrics) {
    benchmarkSummary.textContent = "Held-out test metrics are not included in this model bundle.";
    for (const [, element] of metricElements) element.textContent = "—";
    return;
  }

  benchmarkSummary.textContent = `On a later time period the model detected ${metrics.confusion_matrix[1][1]} of ${metrics.fraud_support} fraud cases, with ${metrics.confusion_matrix[0][1]} false alerts. These results describe this dataset only.`;
  for (const [key, element] of metricElements) {
    const value = Number(metrics[key]);
    element.textContent = key === "precision" || key === "recall" || key === "f1"
      ? `${(value * 100).toFixed(1)}%`
      : value.toFixed(3);
  }
}

function clearBenchmark(message) {
  benchmarkPolicyLabel.textContent = "MODEL METRICS UNAVAILABLE";
  benchmarkSummary.textContent = message;
  for (const selector of [
    "#benchmark-precision",
    "#benchmark-recall",
    "#benchmark-f1",
    "#benchmark-pr-auc",
    "#benchmark-roc-auc",
  ]) {
    document.querySelector(selector).textContent = "—";
  }
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(0)} KiB`;
}

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.hidden = false;
}

function setSelectedFile(file) {
  errorPanel.hidden = true;
  resultsPanel.hidden = true;
  predictionBlob = null;
  downloadButton.disabled = true;

  if (!file) {
    fileTitle.innerHTML = 'Drop your CSV here or <span>browse files</span>';
    fileHint.textContent = "CSV · up to 180 MiB · up to 300,000 rows";
    analyzeButton.disabled = true;
    return;
  }

  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError("Please choose a CSV file.");
    fileInput.value = "";
    analyzeButton.disabled = true;
    return;
  }
  if (file.size > maxUploadBytes) {
    showError("This file is larger than the 180 MiB limit. Choose a smaller CSV.");
    fileInput.value = "";
    analyzeButton.disabled = true;
    return;
  }

  fileTitle.textContent = file.name;
  fileHint.textContent = `${formatBytes(file.size)} · CSV file selected`;
  analyzeButton.disabled = !modelReady || isBusy;
}

function fillPreview(rows) {
  previewBody.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    const values = [
      row.transaction_row,
      Number(row.fraud_score).toFixed(6),
      Number(row.threshold).toFixed(6),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.append(td);
    }
    const labelCell = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = row.predicted_label === "Fraud" ? "label-pill fraud" : "label-pill";
    pill.textContent = row.predicted_label;
    labelCell.append(pill);
    tr.append(labelCell);
    previewBody.append(tr);
  }
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
}
dropzone.addEventListener("drop", (event) => {
  if (isBusy) return;
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file || !modelReady || isBusy) return;

  isBusy = true;
  errorPanel.hidden = true;
  resultsPanel.hidden = true;
  analyzeButton.disabled = true;
  fileInput.disabled = true;
  policySelector.disabled = true;
  analyzeButton.classList.add("is-loading");
  buttonLabel.textContent = "Checking and scoring transactions…";

  try {
    const formData = new FormData();
    formData.append("file", file, file.name);
    formData.append("decision_policy", policySelector.value);
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    if (!response.ok) {
      let message = "The file could not be processed.";
      try {
        const payload = await response.json();
        if (payload.detail) message = payload.detail;
      } catch (_) {
        // Keep the friendly fallback message if the server did not return JSON.
      }
      throw new Error(message);
    }

    predictionBlob = await response.blob();
    const total = Number(response.headers.get("X-Total-Transactions"));
    const flagged = Number(response.headers.get("X-Flagged-Count"));
    const flagRate = Number(response.headers.get("X-Flag-Rate"));
    const previewBase64 = response.headers.get("X-Preview-Base64") || "W10=";
    const base64 = previewBase64.replace(/-/g, "+").replace(/_/g, "/");
    const preview = JSON.parse(atob(base64));

    document.querySelector("#total-count").textContent = total.toLocaleString("en-US");
    document.querySelector("#flagged-count").textContent = flagged.toLocaleString("en-US");
    document.querySelector("#flag-rate").textContent = `${(flagRate * 100).toFixed(2)}%`;
    document.querySelector("#label-warning").hidden = response.headers.get("X-Class-Ignored") !== "true";
    fillPreview(preview);
    resultsPanel.hidden = false;
    downloadButton.disabled = false;
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error instanceof Error ? error.message : "The file could not be processed.");
  } finally {
    isBusy = false;
    fileInput.disabled = false;
    policySelector.disabled = false;
    analyzeButton.disabled = !modelReady || !fileInput.files[0];
    analyzeButton.classList.remove("is-loading");
    buttonLabel.textContent = "Analyze transactions";
  }
});

policySelector.addEventListener("change", updatePolicyDisplay);

downloadButton.addEventListener("click", () => {
  if (!predictionBlob) return;
  const url = URL.createObjectURL(predictionBlob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "fraudshield_predictions.csv";
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
});

modelStatus.textContent = "CHECKING MODEL";
modelStatus.classList.add("status-loading");
fetch("/api/health")
  .then((response) => {
    if (!response.ok) throw new Error("The local model status could not be checked.");
    return response.json();
  })
  .then((health) => {
    if (health.status !== "ready") {
      modelStatus.textContent = "MODEL UNAVAILABLE";
      modelStatus.classList.remove("status-loading");
      modelStatus.classList.add("status-error");
      clearBenchmark("No model is currently available, so model metrics cannot be shown.");
      showError(health.message || "The model could not be loaded. Check the local model file.");
      return;
    }
    if (!Array.isArray(health.decision_policies) || health.decision_policies.length === 0) {
      throw new Error("The local model has no usable review policies.");
    }
    if (!health.decision_policies.some((policy) => policy.id === health.default_policy)) {
      throw new Error("The local model's default review policy is unavailable.");
    }
    modelName.textContent = health.model;
    availablePolicies = new Map(health.decision_policies.map((policy) => [policy.id, policy]));
    policySelector.replaceChildren();
    for (const policy of health.decision_policies) {
      const option = document.createElement("option");
      option.value = policy.id;
      option.textContent = policy.label;
      policySelector.append(option);
    }
    policySelector.value = health.default_policy;
    modelReady = true;
    modelStatus.textContent = "MODEL READY";
    modelStatus.classList.remove("status-loading", "status-error");
    analyzeButton.disabled = !fileInput.files[0];
    updatePolicyDisplay();
  })
  .catch(() => {
    modelStatus.textContent = "MODEL STATUS UNKNOWN";
    modelStatus.classList.remove("status-loading");
    modelStatus.classList.add("status-error");
    modelReady = false;
    analyzeButton.disabled = true;
    clearBenchmark("The local model status could not be confirmed, so model metrics cannot be shown.");
    showError("Could not check the local model. Restart FraudShield AI and try again.");
  });
