from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from config import RVC_FEATURE_DIM, RVC_MODEL_VERSION
from core.domain.audio_asset import AudioAsset
from core.domain.errors import TrainingFailedError
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingJob, TrainingStatus
from core.engines.rvc_inference import run_inference
from core.engines.rvc_layout import ExperimentLayout
from core.engines.rvc_manifest_builder import write_config, write_filelist
from core.engines.rvc_preprocessing import (
    build_train_args,
    run_build_index,
    run_extract_f0,
    run_extract_feature,
    run_preprocess,
    stage_dataset,
)
from infra.audio_io import load_audio_asset
from infra.filesystem_paths import (
    RVC_ASSETS_DIR,
    RVC_DIR,
    RVC_HUBERT_DIR,
    rvc_experiment_dir_for,
    rvc_trained_model_path_for,
)
from infra.job_progress_bus import progress_bus
from infra.process_runner import ProcessRunError, stream_command

_EPOCH_PATTERN = re.compile(r"epoch[:\s]+(\d+)", re.IGNORECASE)


class RvcEngine:
    """VoiceConversionEngine implementation backed by the vendored RVC repo.

    Real RVC training is a 5-stage pipeline (preprocess -> F0 extraction ->
    HuBERT feature extraction -> GAN training -> similarity index), not a
    single command; see rvc_preprocessing.py and rvc_manifest_builder.py
    for the stages that run before the training loop itself. Each training
    run is identified by a caller-supplied model_id, not the voice name,
    since one voice can have several trained models.
    """

    engine_name = "rvc"

    def is_ready(self) -> bool:
        return RVC_DIR.is_dir() and (RVC_HUBERT_DIR / "config.json").is_file()

    def trained_model_path_for(self, model_id: str) -> Path:
        return rvc_trained_model_path_for(model_id)

    def train(self, reference: AudioAsset, config: TrainingConfig, model_id: str) -> TrainingJob:
        job = TrainingJob(
            voice_name=reference.path.stem,
            engine_name=self.engine_name,
            job_id=model_id,
            status=TrainingStatus.RUNNING,
            total_epochs=config.epochs,
            started_at=datetime.utcnow(),
        )
        progress_bus.publish(job)

        try:
            self._run_pipeline(reference, config, job)
        except (ProcessRunError, TrainingFailedError) as exc:
            job.status = TrainingStatus.FAILED
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            progress_bus.publish(job)
            progress_bus.close(job.job_id)
            raise TrainingFailedError(f"RVC training failed: {exc}") from exc

        job.status = TrainingStatus.COMPLETED
        job.model_output_path = rvc_trained_model_path_for(job.job_id)
        job.finished_at = datetime.utcnow()
        progress_bus.publish(job)
        progress_bus.close(job.job_id)
        return job

    def _run_pipeline(
        self, reference: AudioAsset, config: TrainingConfig, job: TrainingJob
    ) -> None:
        model_id = job.job_id
        exp_dir = rvc_experiment_dir_for(model_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        # train/process_ckpt.py's savee() writes here with a bare relative
        # path and doesn't create it itself - torch.save fails otherwise.
        rvc_trained_model_path_for(model_id).parent.mkdir(parents=True, exist_ok=True)
        layout = ExperimentLayout(exp_dir, RVC_FEATURE_DIM)

        def on_line(line: str) -> None:
            self._on_line(job, line)

        dataset_dir = stage_dataset(reference.path, exp_dir)
        run_preprocess(dataset_dir, exp_dir, config.sample_rate, on_line)
        run_extract_f0(exp_dir, on_line)
        run_extract_feature(exp_dir, on_line=on_line)

        write_filelist(layout, config.sample_rate, RVC_FEATURE_DIM)
        write_config(layout, config.sample_rate, RVC_MODEL_VERSION)

        self._run_train_loop(job, config)

        if config.use_similarity_index:
            run_build_index(model_id, on_line=on_line)

    def _run_train_loop(self, job: TrainingJob, config: TrainingConfig) -> None:
        args = build_train_args(job.job_id, config)
        for line in stream_command(args, cwd=RVC_DIR):
            self._on_line(job, line)

    def convert(self, source_audio: AudioAsset, trained_model_path: Path) -> AudioAsset:
        model_id = trained_model_path.stem
        output_path = trained_model_path.parent / f"{model_id}_{source_audio.path.stem}.wav"
        result_path = run_inference(source_audio.path, trained_model_path, output_path)
        return load_audio_asset(result_path)

    def delete_trained_model(self, model_id: str) -> None:
        rvc_trained_model_path_for(model_id).unlink(missing_ok=True)
        shutil.rmtree(rvc_experiment_dir_for(model_id), ignore_errors=True)
        for index_path in (RVC_ASSETS_DIR / "indices").glob(f"{model_id}_*"):
            index_path.unlink(missing_ok=True)

    def _on_line(self, job: TrainingJob, line: str) -> None:
        job.append_log_line(line)
        match = _EPOCH_PATTERN.search(line)
        if match:
            job.current_epoch = int(match.group(1))
        progress_bus.publish(job)
