from __future__ import annotations

import shutil
import sys
from pathlib import Path

from config import RVC_MODEL_VERSION
from core.domain.errors import TrainingFailedError
from core.domain.training_config import TrainingConfig
from core.engines.rvc_layout import sample_rate_label
from infra.filesystem_paths import RVC_ASSETS_DIR, RVC_DIR, RVC_PRETRAINED_DIR
from infra.process_runner import ProcessRunError, run_command

_PREPROCESS_SEGMENT_SECONDS = 3.7
_GPU_INDEX = "0"


def stage_dataset(reference_path: Path, exp_dir: Path) -> Path:
    """Copy the single reference clip into its own dataset dir.

    preprocess.py slices every audio file found in the directory it's
    given, so that directory must contain only this training run's audio.
    """
    dataset_dir = exp_dir / "_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference_path, dataset_dir / reference_path.name)
    return dataset_dir


def _run_module(module: str, args: list[str], step_name: str) -> None:
    # Invoked as "-m package.module", not "python path/to/module.py": some
    # of these packages (e.g. train/) contain a file named the same as the
    # package itself (train/train.py), which shadows the real package on
    # sys.path[0] when run by file path. -m instead puts cwd (RVC_DIR)
    # first on sys.path, avoiding the collision.
    try:
        run_command([sys.executable, "-m", module, *args], cwd=RVC_DIR)
    except ProcessRunError as exc:
        raise TrainingFailedError(f"RVC {step_name} failed: {exc}") from exc


def run_preprocess(dataset_dir: Path, exp_dir: Path, sample_rate: int) -> None:
    _run_module(
        "train.preprocess",
        [
            str(dataset_dir),
            str(sample_rate),
            "1",
            str(exp_dir),
            "True",
            str(_PREPROCESS_SEGMENT_SECONDS),
        ],
        "preprocessing",
    )


def run_extract_f0(exp_dir: Path) -> None:
    _run_module(
        "train.dataset.extract_f0",
        ["cuda", "1", "0", _GPU_INDEX, str(exp_dir), "True"],
        "F0 extraction",
    )


def run_extract_feature(exp_dir: Path, version: str = RVC_MODEL_VERSION) -> None:
    _run_module(
        "train.dataset.extract_hubert_feature",
        ["cuda:0", "1", "0", _GPU_INDEX, str(exp_dir), version, "True"],
        "feature extraction",
    )


def run_build_index(voice_name: str, version: str = RVC_MODEL_VERSION) -> None:
    indices_dir = RVC_ASSETS_DIR / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    _run_module(
        "train.train_index",
        [voice_name, version, str(indices_dir), "4"],
        "index building",
    )


def build_train_args(voice_name: str, config: TrainingConfig) -> list[str]:
    sr_label = sample_rate_label(config.sample_rate)
    pretrained_g = RVC_PRETRAINED_DIR / f"f0G{sr_label}.pth"
    pretrained_d = RVC_PRETRAINED_DIR / f"f0D{sr_label}.pth"
    return [
        sys.executable, "-m", "train.train",
        "-e", voice_name,
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
