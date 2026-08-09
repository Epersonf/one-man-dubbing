from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrainedModelInfo:
    """Metadata for one trained model. A voice can have several of these -
    e.g. re-trained with more epochs, or with a different engine.
    """

    model_id: str
    voice_name: str
    engine_name: str
    created_at: datetime
    epochs: int
    has_similarity_index: bool
