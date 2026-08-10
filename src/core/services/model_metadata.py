from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from core.domain.trained_model import TrainedModelInfo
from core.domain.training_config import TrainingConfig
from core.registry.engine_registry import EngineRegistry
from infra.filesystem_paths import model_metadata_path_for
from infra.job_progress_bus import progress_bus
from infra.metadata_store import read_json, write_json


def new_model_id(voice_name: str) -> str:
    return f"{voice_name}__{uuid4().hex[:8]}"


def save_if_usable(model_id: str, voice_name: str, config: TrainingConfig, engine_name: str) -> None:
    """Persist model metadata as long as the engine has a usable weights
    file for it - even if the training run that produced it later failed
    or is still in progress. That's what makes a model trained to epoch
    500 of a 2000-epoch target show up as a resumable checkpoint instead
    of a lost run.
    """
    engine = EngineRegistry.get_conversion(engine_name)
    if not engine.trained_model_path_for(model_id).is_file():
        return

    job = progress_bus.latest_for_job(model_id)
    epochs_completed = job.current_epoch if job else config.epochs
    existing = read_json(model_metadata_path_for(voice_name, model_id))
    created_at = existing["created_at"] if existing else datetime.utcnow().isoformat()

    write_json(
        model_metadata_path_for(voice_name, model_id),
        {
            "model_id": model_id,
            "voice_name": voice_name,
            "engine_name": engine_name,
            "created_at": created_at,
            "epochs": config.epochs,
            "epochs_completed": max(epochs_completed, (existing or {}).get("epochs_completed", 0)),
            "sample_rate": config.sample_rate,
            "has_similarity_index": config.use_similarity_index,
        },
    )


def from_metadata(data: dict) -> TrainedModelInfo:
    return TrainedModelInfo(
        model_id=data["model_id"],
        voice_name=data["voice_name"],
        engine_name=data["engine_name"],
        created_at=datetime.fromisoformat(data["created_at"]),
        epochs=data["epochs"],
        epochs_completed=data.get("epochs_completed", data["epochs"]),
        sample_rate=data.get("sample_rate", 40_000),
        has_similarity_index=data["has_similarity_index"],
    )
