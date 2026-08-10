from __future__ import annotations

import sys

from config import RVC_MODEL_VERSION
from core.domain.training_config import TrainingConfig
from core.engines.rvc_layout import sample_rate_label
from infra.filesystem_paths import RVC_DIR, RVC_PRETRAINED_DIR


def build_train_args(model_id: str, config: TrainingConfig) -> list[str]:
    sr_label = sample_rate_label(config.sample_rate)
    pretrained_g = RVC_PRETRAINED_DIR / f"f0G{sr_label}.pth"
    pretrained_d = RVC_PRETRAINED_DIR / f"f0D{sr_label}.pth"
    return [
        sys.executable, "-m", "train.train",
        "-e", model_id,
        "-sr", sr_label,
        "-f0", "1",
        "-bs", str(config.batch_size),
        "-g", "0",
        "-te", str(config.epochs),
        "-se", str(config.save_every_epoch),
        "-pg", str(pretrained_g.relative_to(RVC_DIR)),
        "-pd", str(pretrained_d.relative_to(RVC_DIR)),
        "-l", "1",
        "-c", "0",
        "-sw", "1",
        "-v", RVC_MODEL_VERSION,
    ]
