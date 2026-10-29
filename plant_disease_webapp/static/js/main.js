/**
 * main.js — Plant Disease Detection Web App
 *
 * Handles:
 *  - File selection (click + drag-and-drop)
 *  - Image preview
 *  - AJAX POST to /predict
 *  - Loading state
 *  - Result display (success + error)
 *  - All-probabilities toggle
 */

(function () {
  "use strict";

  // ── DOM references ──────────────────────────────────────────────────────
  const uploadZone     = document.getElementById("upload-zone");
  const fileInput      = document.getElementById("file-input");
  const previewSection = document.getElementById("preview-section");
  const imagePreview   = document.getElementById("image-preview");
  const fileInfoEl     = document.getElementById("file-info");
  const changeBtn      = document.getElementById("change-btn");
  const predictBtn     = document.getElementById("predict-btn");
  const loadingSection = document.getElementById("loading-section");
  const resultSection  = document.getElementById("result-section");
  const errorSection   = document.getElementById("error-section");
  const errorMsg       = document.getElementById("error-msg");
  const probsToggle    = document.getElementById("probs-toggle");
  const probsTable     = document.getElementById("probs-table");

  let selectedFile = null;

  // ── File selection via click ────────────────────────────────────────────
  fileInput.addEventListener("change", function () {
    if (this.files && this.files[0]) {
      handleFile(this.files[0]);
    }
  });

  // ── Drag-and-drop ────────────────────────────────────────────────────────
  uploadZone.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", function () {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // ── Change button (re-upload) ───────────────────────────────────────────
  changeBtn.addEventListener("click", function () {
    fileInput.value = "";
    selectedFile    = null;
    previewSection.classList.remove("visible");
    predictBtn.disabled = true;
    hideResults();
    fileInput.click();
  });

  // ── Predict button ───────────────────────────────────────────────────────
  predictBtn.addEventListener("click", runPrediction);

  // ── Handle a selected/dropped file ──────────────────────────────────────
  function handleFile(file) {
    // Validate type
    const allowed = ["image/jpeg", "image/png", "image/jpg"];
    if (!allowed.includes(file.type)) {
      showError("Unsupported file type. Please upload a JPG or PNG image.");
      return;
    }

    // Validate size (16 MB)
    if (file.size > 16 * 1024 * 1024) {
      showError("File is too large. Maximum size is 16 MB.");
      return;
    }

    selectedFile = file;
    hideResults();

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      previewSection.classList.add("visible");
    };
    reader.readAsDataURL(file);

    // Show file info
    fileInfoEl.textContent =
      file.name + " · " + formatFileSize(file.size);

    predictBtn.disabled = false;
  }

  // ── Run prediction via AJAX ──────────────────────────────────────────────
  function runPrediction() {
    if (!selectedFile) return;

    hideResults();
    showLoading(true);
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    fetch("/predict", {
      method : "POST",
      body   : formData,
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, status: res.status, data: data };
        });
      })
      .then(function (resp) {
        showLoading(false);
        predictBtn.disabled = false;

        if (resp.ok && resp.data.success) {
          showResult(resp.data);
        } else {
          const msg = (resp.data && resp.data.error)
            ? resp.data.error
            : "Prediction failed (status " + resp.status + ").";
          showError(msg);
        }
      })
      .catch(function (err) {
        showLoading(false);
        predictBtn.disabled = false;
        showError(
          "Network error — could not reach the server. " +
          "Please check your connection and try again."
        );
        console.error("Fetch error:", err);
      });
  }

  // ── Display result ───────────────────────────────────────────────────────
  function showResult(data) {
    // Emoji + disease name
    document.getElementById("result-emoji").textContent   = data.emoji   || "🌿";
    document.getElementById("result-disease").textContent = data.display || data.disease;

    // Confidence bar
    const pct = Math.min(Math.max(data.confidence || 0), 100);
    document.getElementById("confidence-pct").textContent = pct.toFixed(1) + "%";
    // Animate bar width after a short delay so transition fires
    setTimeout(function () {
      document.getElementById("bar-fill").style.width = pct + "%";
    }, 50);

    // Severity badge
    const badgeEl  = document.getElementById("severity-badge");
    const severity = (data.severity || "unknown").toLowerCase();
    badgeEl.textContent = data.severity || "Unknown";
    badgeEl.className   = "severity-badge severity-" + severity;

    // Description and action
    document.getElementById("result-description").textContent =
      data.description || "";
    document.getElementById("result-action").textContent =
      data.action || "";

    // All probabilities table
    buildProbsTable(data.all_probs || {});

    // Show
    resultSection.classList.add("visible");

    // Smooth scroll to result
    setTimeout(function () {
      resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }

  // ── Build probability breakdown table ────────────────────────────────────
  function buildProbsTable(probs) {
    probsTable.innerHTML = "";
    const entries = Object.entries(probs).sort(function (a, b) {
      return b[1] - a[1];
    });
    entries.forEach(function (entry) {
      const row      = document.createElement("div");
      row.className  = "prob-row";
      const nameSpan = document.createElement("span");
      nameSpan.className   = "prob-name";
      nameSpan.textContent = entry[0];
      const valSpan  = document.createElement("span");
      valSpan.className    = "prob-val";
      valSpan.textContent  = entry[1].toFixed(1) + "%";
      row.appendChild(nameSpan);
      row.appendChild(valSpan);
      probsTable.appendChild(row);
    });
    // Reset toggle state
    probsTable.classList.remove("open");
    probsToggle.textContent = "▶ Show all probabilities";
  }

  // ── Probabilities toggle ─────────────────────────────────────────────────
  probsToggle.addEventListener("click", function () {
    const isOpen = probsTable.classList.toggle("open");
    probsToggle.textContent = isOpen
      ? "▲ Hide probabilities"
      : "▶ Show all probabilities";
  });

  // ── Show error ────────────────────────────────────────────────────────────
  function showError(msg) {
    errorMsg.textContent = msg;
    errorSection.classList.add("visible");
    errorSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ── Loading state ─────────────────────────────────────────────────────────
  function showLoading(on) {
    if (on) loadingSection.classList.add("visible");
    else    loadingSection.classList.remove("visible");
  }

  // ── Hide all result panels ────────────────────────────────────────────────
  function hideResults() {
    resultSection.classList.remove("visible");
    errorSection.classList.remove("visible");
    loadingSection.classList.remove("visible");
    // Reset confidence bar
    document.getElementById("bar-fill").style.width = "0%";
  }

  // ── Utility ────────────────────────────────────────────────────────────
  function formatFileSize(bytes) {
    if (bytes < 1024)        return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

})();
