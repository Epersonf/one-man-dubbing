from __future__ import annotations

from math import gcd
from pathlib import Path

from core.domain.audio_asset import AudioAsset
from core.domain.errors import AudioValidationError

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg"})

# Formats whose encoders only accept specific sample rates. RVC's own
# working rate (e.g. 40000 Hz) isn't MP3-legal, so audio produced at that
# rate has to be resampled before MP3 encoding or libsndfile rejects it.
_FORMAT_VALID_SAMPLE_RATES: dict[str, frozenset[int]] = {
    "mp3": frozenset({8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}),
}
_FALLBACK_SAMPLE_RATE = 44100


def load_audio_asset(path: Path) -> AudioAsset:
    """Validate an audio file on disk and describe it as an AudioAsset.

    Single entry point for reading audio metadata; nothing outside infra/
    should touch soundfile/librosa directly (DRY + layering).
    """
    if not path.is_file():
        raise AudioValidationError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(f"Unsupported audio format: {path.suffix}")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioValidationError("soundfile is not installed") from exc

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise AudioValidationError(f"Could not read audio file {path}: {exc}") from exc

    return AudioAsset(
        path=path,
        duration_seconds=info.frames / info.samplerate if info.samplerate else 0.0,
        sample_rate=info.samplerate,
    )


def save_uploaded_audio(raw_bytes: bytes, destination: Path) -> AudioAsset:
    """Persist uploaded bytes to disk and return the validated AudioAsset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw_bytes)
    return load_audio_asset(destination)


def convert_to_format(source: AudioAsset, target_path: Path, target_format: str = "mp3") -> AudioAsset:
    """Convert an audio asset to the requested output format via soundfile."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioValidationError("soundfile is not installed") from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data, samplerate = sf.read(str(source.path))
        valid_rates = _FORMAT_VALID_SAMPLE_RATES.get(target_format.lower())
        if valid_rates is not None and samplerate not in valid_rates:
            data, samplerate = _resample(data, samplerate, _FALLBACK_SAMPLE_RATE)
        sf.write(str(target_path), data, samplerate, format=target_format.upper())
    except Exception as exc:
        raise AudioValidationError(f"Could not convert audio to {target_format}: {exc}") from exc

    return load_audio_asset(target_path)


def _resample(data, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return data, source_rate
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    resampled = resample_poly(data, target_rate // divisor, source_rate // divisor, axis=0)
    return resampled, target_rate
