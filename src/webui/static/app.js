function watchTrainingProgress(voiceName) {
    const progressPanel = document.getElementById("progress-panel");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const continueLink = document.getElementById("continue-link");

    progressPanel.hidden = false;
    const source = new EventSource(`/step2/progress/${encodeURIComponent(voiceName)}`);

    source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        progressBar.value = data.progress_ratio;
        progressText.textContent = `${data.status.toUpperCase()} — epoch ${data.current_epoch}/${data.total_epochs}`;

        if (data.status === "completed") {
            continueLink.hidden = false;
            source.close();
        }
        if (data.status === "failed") {
            progressText.textContent = `FAILED: ${data.error_message || "unknown error"}`;
            source.close();
        }
    };

    source.onerror = () => {
        source.close();
    };
}

document.addEventListener("DOMContentLoaded", () => {
    const trainForm = document.getElementById("train-form");
    if (!trainForm) {
        return;
    }

    trainForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const voiceName = trainForm.dataset.voiceName;
        const formData = new FormData(trainForm);
        formData.set("voice_name", voiceName);
        formData.set("use_similarity_index", trainForm.use_similarity_index.checked ? "true" : "false");

        await fetch("/step2/train", { method: "POST", body: formData });
        watchTrainingProgress(voiceName);
    });
});
