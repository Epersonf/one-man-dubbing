function watchTrainingProgress(modelId, submitButton) {
    const progressPanel = document.getElementById("progress-panel");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const progressLog = document.getElementById("progress-log");
    const continueLink = document.getElementById("continue-link");

    progressPanel.hidden = false;
    const source = new EventSource(`/step2/progress/${encodeURIComponent(modelId)}`);

    const finish = () => {
        source.close();
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "START TRAINING";
        }
    };

    source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        progressBar.value = data.progress_ratio;
        progressText.textContent = `${data.status.toUpperCase()} — epoch ${data.current_epoch}/${data.total_epochs}`;
        if (data.log_lines && data.log_lines.length) {
            progressLog.textContent = data.log_lines.join("\n");
            progressLog.scrollTop = progressLog.scrollHeight;
        }

        if (data.status === "completed") {
            continueLink.hidden = false;
            finish();
        }
        if (data.status === "failed") {
            progressText.textContent = `FAILED: ${data.error_message || "unknown error"}`;
            finish();
        }
    };

    source.onerror = finish;
}

function setupTrainForm() {
    const trainForm = document.getElementById("train-form");
    if (!trainForm) {
        return;
    }
    const submitButton = document.getElementById("train-submit");

    trainForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (submitButton.disabled) {
            return;
        }
        submitButton.disabled = true;
        submitButton.textContent = "TRAINING...";

        const voiceName = trainForm.dataset.voiceName;
        const formData = new FormData(trainForm);
        formData.set("voice_name", voiceName);
        formData.set("use_similarity_index", trainForm.use_similarity_index.checked ? "true" : "false");

        const response = await fetch("/step2/train", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) {
            document.getElementById("progress-text").textContent = `FAILED: ${result.error || "could not start training"}`;
            document.getElementById("progress-panel").hidden = false;
            submitButton.disabled = false;
            submitButton.textContent = "START TRAINING";
            return;
        }
        watchTrainingProgress(result.model_id, submitButton);
    });
}

function setupDeleteConfirmations() {
    document.querySelectorAll("form.inline-form[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (!window.confirm(form.dataset.confirm)) {
                event.preventDefault();
            }
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupTrainForm();
    setupDeleteConfirmations();
});
