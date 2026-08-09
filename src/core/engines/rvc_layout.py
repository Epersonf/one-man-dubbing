from __future__ import annotations

from pathlib import Path

from config import RVC_FEATURE_DIM

_SAMPLE_RATE_LABELS: dict[int, str] = {32_000: "32k", 40_000: "40k", 48_000: "48k"}


def sample_rate_label(sample_rate: int) -> str:
    try:
        return _SAMPLE_RATE_LABELS[sample_rate]
    except KeyError as exc:
        raise ValueError(f"Unsupported RVC sample rate: {sample_rate}") from exc


class ExperimentLayout:
    """RVC's own internal directory layout for one training run, rooted at
    vendor/rvc/logs/<voice_name>/ (populated by preprocess.py, extract_f0.py
    and extract_hubert_feature.py before training starts).
    """

    def __init__(self, exp_dir: Path, feature_dim: int = RVC_FEATURE_DIM) -> None:
        self.exp_dir = exp_dir
        self.gt_wavs_dir = exp_dir / "0_gt_wavs"
        self.f0_dir = exp_dir / "2a_f0"
        self.f0nsf_dir = exp_dir / "2b-f0nsf"
        self.feature_dir = exp_dir / f"3_feature{feature_dim}"
        self.filelist_path = exp_dir / "filelist.txt"
        self.config_path = exp_dir / "config.json"
