from __future__ import annotations

import json
from random import shuffle

from core.domain.errors import TrainingFailedError
from core.engines.rvc_layout import ExperimentLayout, sample_rate_label
from infra.filesystem_paths import RVC_DIR, RVC_MUTE_DIR

_SPEAKER_ID = 0


def _escaped(path_str: str) -> str:
    # Matches upstream RVC's own filelist formatting (webui.py); harmless
    # doubling of separators, tolerated by both Windows and POSIX I/O.
    return path_str.replace("\\", "\\\\")


def _sample_names(layout: ExperimentLayout) -> list[str]:
    def stems(directory):
        if not directory.is_dir():
            return set()
        return {entry.name.split(".")[0] for entry in directory.iterdir()}

    names = (
        stems(layout.gt_wavs_dir)
        & stems(layout.feature_dir)
        & stems(layout.f0_dir)
        & stems(layout.f0nsf_dir)
    )
    if not names:
        raise TrainingFailedError(
            "No preprocessed samples found — preprocessing or feature "
            "extraction likely failed or produced no output."
        )
    return sorted(names)


def _sample_line(layout: ExperimentLayout, name: str) -> str:
    return "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
        _escaped(str(layout.gt_wavs_dir)),
        name,
        _escaped(str(layout.feature_dir)),
        name,
        _escaped(str(layout.f0_dir)),
        name,
        _escaped(str(layout.f0nsf_dir)),
        name,
        _SPEAKER_ID,
    )


def _mute_lines(sr_label: str, feature_dim: int) -> list[str]:
    gt = RVC_MUTE_DIR / "0_gt_wavs" / f"mute{sr_label}.wav"
    feature = RVC_MUTE_DIR / f"3_feature{feature_dim}" / "mute.npy"
    f0 = RVC_MUTE_DIR / "2a_f0" / "mute.wav.npy"
    f0nsf = RVC_MUTE_DIR / "2b-f0nsf" / "mute.wav.npy"
    line = f"{gt}|{feature}|{f0}|{f0nsf}|{_SPEAKER_ID}"
    return [line, line]


def write_filelist(layout: ExperimentLayout, sample_rate: int, feature_dim: int) -> None:
    lines = [_sample_line(layout, name) for name in _sample_names(layout)]
    lines.extend(_mute_lines(sample_rate_label(sample_rate), feature_dim))
    shuffle(lines)
    layout.filelist_path.write_text("\n".join(lines), encoding="utf8")


def write_config(layout: ExperimentLayout, sample_rate: int, version: str) -> None:
    sr_label = sample_rate_label(sample_rate)
    template_group = "v1" if version == "v1" or sr_label == "40k" else "v2"
    template_path = RVC_DIR / "configs" / template_group / f"{sr_label}.json"
    config_data = json.loads(template_path.read_text(encoding="utf8"))
    config_data.pop("speaker_info", None)
    layout.config_path.write_text(
        json.dumps(config_data, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf8",
    )
