from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrainedModelInfo:
    """Metadata for one trained model. A voice can have several of these -
    e.g. re-trained with more epochs, or with a different engine.

    A model becomes visible/usable as soon as the engine has written its
    first checkpoint (see TrainingService._save_model_metadata_if_usable),
    even if the run later fails or is still in progress - that's what
    makes epochs_completed < epochs a meaningful, resumable state rather
    than just a lost run.
    """

    model_id: str
    voice_name: str
    engine_name: str
    created_at: datetime
    epochs: int
    epochs_completed: int
    sample_rate: int
    has_similarity_index: bool

    @property
    def is_complete(self) -> bool:
        return self.epochs_completed >= self.epochs
