from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from core.domain.audio_asset import AudioAsset
from core.domain.errors import TrainingFailedError
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob, TrainingStatus
from core.engines.rvc_inference import run_inference
from infra.audio_io import load_audio_asset
from infra.filesystem_paths import RVC_DIR, WEIGHTS_DIR, model_dir_for
from infra.job_progress_bus import progress_bus
from infra.process_runner import ProcessRunError, stream_command

_EPOCH_PATTERN = re.compile(r"epoch[:\s]+(\d+)", re.IGNORECASE)


class RvcEngine:
    """VoiceConversionEngine implementation backed by the vendored RVC repo."""

    engine_name = "rvc"

    def is_ready(self) -> bool:
        hubert_weights = WEIGHTS_DIR / "rvc" / "hubert_base.pt"
        return RVC_DIR.is_dir() and hubert_weights.is_file()

    def train(self, reference: AudioAsset, config: TrainingConfig) -> TrainingJob:
        job = TrainingJob(
            voice_name=reference.path.stem,
            engine_name=self.engine_name,
            status=TrainingStatus.RUNNING,
            total_epochs=config.epochs,
            started_at=datetime.utcnow(),
        )
        output_dir = model_dir_for(job.voice_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_bus.publish(job)

        training_script = RVC_DIR / "tools" / "train_cli.py"
        args = [
            "python",
            str(training_script),
            "--reference",
            str(reference.path),
            "--epochs",
            str(config.epochs),
            "--sample-rate",
            str(config.sample_rate),
            "--batch-size",
            str(config.batch_size),
            "--output-dir",
            str(output_dir),
        ]
        if config.use_similarity_index:
            args.append("--use-index")

        try:
            for line in stream_command(args, cwd=RVC_DIR):
                self._apply_progress_line(job, line)
        except ProcessRunError as exc:
            job.status = TrainingStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            progress_bus.publish(job)
            progress_bus.close(job.job_id)
            raise TrainingFailedError(f"RVC training failed: {exc}") from exc

        job.status = TrainingStatus.COMPLETED
        job.model_output_path = output_dir / "model.pth"
        job.finished_at = datetime.utcnow()
        progress_bus.publish(job)
        progress_bus.close(job.job_id)
        return job

    def convert(self, source_audio: AudioAsset, trained_model_path: Path) -> AudioAsset:
        output_path = trained_model_path.parent / "converted_output.wav"
        result_path = run_inference(source_audio.path, trained_model_path, output_path)
        return load_audio_asset(result_path)

    def _apply_progress_line(self, job: TrainingJob, line: str) -> None:
        match = _EPOCH_PATTERN.search(line)
        if match:
            job.current_epoch = int(match.group(1))
            progress_bus.publish(job)
