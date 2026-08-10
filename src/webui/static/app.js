function watchTrainingProgress(modelId, submitButton, resetLabel) {
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
            submitButton.textContent = resetLabel || "START TRAINING";
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

function setupResumeForms() {
    document.querySelectorAll("form.resume-form").forEach((form) => {
        const submitButton = form.querySelector("button[type=submit]");
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (submitButton.disabled) {
                return;
            }
            const originalLabel = submitButton.textContent;
            submitButton.disabled = true;
            submitButton.textContent = "RESUMING...";

            const formData = new FormData(form);
            formData.set("voice_name", form.dataset.voiceName);
            formData.set("model_id", form.dataset.modelId);

            const response = await fetch("/step2/resume", { method: "POST", body: formData });
            const result = await response.json();
            if (!response.ok) {
                document.getElementById("progress-text").textContent = `FAILED: ${result.error || "could not resume training"}`;
                document.getElementById("progress-panel").hidden = false;
                submitButton.disabled = false;
                submitButton.textContent = originalLabel;
                return;
            }
            watchTrainingProgress(result.model_id, submitButton, originalLabel);
        });
    });
}

function watchDubbingProgress(jobId, voiceName, outputFormat, submitButton) {
    const progressPanel = document.getElementById("dub-progress-panel");
    const progressText = document.getElementById("dub-progress-text");
    const progressLog = document.getElementById("dub-progress-log");
    const downloadLink = document.getElementById("dub-download-link");

    progressPanel.hidden = false;
    const query = `voice_name=${encodeURIComponent(voiceName)}&output_format=${encodeURIComponent(outputFormat)}`;
    const source = new EventSource(`/step3/progress/${encodeURIComponent(jobId)}?${query}`);

    const finish = () => {
        source.close();
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "GENERATE DUBBED AUDIO";
        }
    };

    source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        progressText.textContent = data.status.toUpperCase();
        if (data.log_lines && data.log_lines.length) {
            progressLog.textContent = data.log_lines.join("\n");
            progressLog.scrollTop = progressLog.scrollHeight;
        }

        if (data.status === "completed") {
            downloadLink.href = data.download_url;
            downloadLink.hidden = false;
            finish();
        }
        if (data.status === "failed") {
            progressText.textContent = `FAILED: ${data.error_message || "unknown error"}`;
            finish();
        }
    };

    source.onerror = finish;
}

function setupDubForm() {
    const dubForm = document.getElementById("dub-form");
    if (!dubForm) {
        return;
    }
    const submitButton = document.getElementById("dub-submit");

    dubForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (submitButton.disabled) {
            return;
        }
        submitButton.disabled = true;
        submitButton.textContent = "DUBBING...";

        const voiceName = dubForm.dataset.voiceName;
        const formData = new FormData(dubForm);
        formData.set("voice_name", voiceName);
        const outputFormat = formData.get("output_format");

        const response = await fetch("/step3/dub", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) {
            document.getElementById("dub-progress-text").textContent = `FAILED: ${result.error || "could not start dubbing"}`;
            document.getElementById("dub-progress-panel").hidden = false;
            submitButton.disabled = false;
            submitButton.textContent = "GENERATE DUBBED AUDIO";
            return;
        }
        watchDubbingProgress(result.job_id, voiceName, outputFormat, submitButton);
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
    setupResumeForms();
    setupDubForm();
    setupDeleteConfirmations();
});
