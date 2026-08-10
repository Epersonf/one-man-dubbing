from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from config import RVC_MODEL_VERSION
from core.domain.errors import TrainingFailedError
from core.domain.training_config import TrainingConfig
from core.engines.rvc_layout import ExperimentLayout
from core.engines.rvc_manifest_builder import write_config, write_filelist
from infra.filesystem_paths import RVC_ASSETS_DIR, RVC_DIR
from infra.process_runner import ProcessRunError, stream_command

_PREPROCESS_SEGMENT_SECONDS = 3.7
_GPU_INDEX = "0"
OnLine = Callable[[str], None]


def stage_dataset(reference_paths: list[Path], exp_dir: Path) -> Path:
    """Copy every reference clip into its own dataset dir.

    preprocess.py slices every audio file found in the directory it's
    given, so that directory must contain only this training run's audio.
    Clips are index-prefixed so same-named uploads (e.g. two "clip.wav"
    from different folders) never collide.
    """
    dataset_dir = exp_dir / "_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(reference_paths):
        shutil.copyfile(path, dataset_dir / f"{index:02d}_{path.name}")
    return dataset_dir


def _run_module(module: str, args: list[str], step_name: str, on_line: OnLine | None = None) -> None:
    # Invoked as "-m package.module", not "python path/to/module.py": some
    # of these packages (e.g. train/) contain a file named the same as the
    # package itself (train/train.py), which shadows the real package on
    # sys.path[0] when run by file path. -m instead puts cwd (RVC_DIR)
    # first on sys.path, avoiding the collision.
    try:
        for line in stream_command([sys.executable, "-m", module, *args], cwd=RVC_DIR):
            if on_line is not None:
                on_line(line)
    except ProcessRunError as exc:
        raise TrainingFailedError(f"RVC {step_name} failed: {exc}") from exc


def run_preprocess(
    dataset_dir: Path, exp_dir: Path, sample_rate: int, on_line: OnLine | None = None
) -> None:
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
        on_line,
    )


def run_preprocessing_stages(
    reference_paths: list[Path],
    exp_dir: Path,
    config: TrainingConfig,
    feature_dim: int,
    on_line: OnLine | None = None,
) -> None:
    """Runs stage_dataset + preprocess.py + F0/feature extraction, then
    writes the training filelist/config - everything before the GAN
    training loop. Skipped entirely when resuming a previous run (see
    RvcEngine.train's resume flag): that data is already on disk.
    """
    layout = ExperimentLayout(exp_dir, feature_dim)
    dataset_dir = stage_dataset(reference_paths, exp_dir)
    run_preprocess(dataset_dir, exp_dir, config.sample_rate, on_line)
    run_extract_f0(exp_dir, on_line)
    run_extract_feature(exp_dir, on_line=on_line)
    write_filelist(layout, config.sample_rate, feature_dim)
    write_config(layout, config.sample_rate, RVC_MODEL_VERSION)


def run_extract_f0(exp_dir: Path, on_line: OnLine | None = None) -> None:
    _run_module(
        "train.dataset.extract_f0",
        ["cuda", "1", "0", _GPU_INDEX, str(exp_dir), "True"],
        "F0 extraction",
        on_line,
    )


def run_extract_feature(
    exp_dir: Path, version: str = RVC_MODEL_VERSION, on_line: OnLine | None = None
) -> None:
    _run_module(
        "train.dataset.extract_hubert_feature",
        ["cuda:0", "1", "0", _GPU_INDEX, str(exp_dir), version, "True"],
        "feature extraction",
        on_line,
    )


def run_build_index(
    model_id: str, version: str = RVC_MODEL_VERSION, on_line: OnLine | None = None
) -> None:
    indices_dir = RVC_ASSETS_DIR / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    _run_module(
        "train.train_index",
        [model_id, version, str(indices_dir), "4"],
        "index building",
        on_line,
    )
