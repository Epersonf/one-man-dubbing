from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for training a voice conversion model.

    Defaults are sane for a 12 GB RTX 3060.
    """

    epochs: int = 200
    sample_rate: int = 40_000
    batch_size: int = 8
    use_similarity_index: bool = True
    save_every_epoch: int = 25

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
