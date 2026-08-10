from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime

from core.domain.voice_profile import VoiceProfile, VoiceSourceRoute
from infra.filesystem_paths import (
    REFERENCES_DIR,
    legacy_reference_path_for,
    reference_dir_for,
    voice_metadata_path_for,
)

SaveMetadata = Callable[[VoiceProfile], None]


def migrate_legacy_single_file(voice_name: str) -> None:
    """Voices created before multi-clip support have a single
    data/references/<name>.wav instead of a data/references/<name>/
    folder - move it into the new layout in place.
    """
    legacy_path = legacy_reference_path_for(voice_name)
    clip_dir = reference_dir_for(voice_name)
    if legacy_path.is_file() and not clip_dir.is_dir():
        clip_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_path), str(clip_dir / f"clip_00{legacy_path.suffix}"))


def backfill_missing_metadata(save_metadata: SaveMetadata) -> None:
    """Voices with clips on disk but no metadata sidecar (created before
    sidecars existed, or dropped in by hand) get one generated so they
    show up like any other voice.
    """
    for legacy_wav in REFERENCES_DIR.glob("*.wav"):
        migrate_legacy_single_file(legacy_wav.stem)
    for clip_dir in REFERENCES_DIR.iterdir():
        if clip_dir.is_dir() and not voice_metadata_path_for(clip_dir.name).is_file():
            backfill_metadata_for(clip_dir.name, save_metadata)


def backfill_metadata_for(voice_name: str, save_metadata: SaveMetadata) -> None:
    clip_dir = reference_dir_for(voice_name)
    save_metadata(
        VoiceProfile(
            name=voice_name,
            reference_dir=clip_dir,
            source_route=VoiceSourceRoute.UPLOAD,
            created_at=datetime.fromtimestamp(clip_dir.stat().st_mtime),
        )
    )
