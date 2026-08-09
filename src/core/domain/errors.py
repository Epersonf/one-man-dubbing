from __future__ import annotations


class OneManDubbingError(Exception):
    """Base class for all domain-level errors.

    Infrastructure and engine code must catch third-party exceptions and
    re-raise one of these instead, so no external stack trace ever reaches
    the UI.
    """


class EngineNotReadyError(OneManDubbingError):
    """Raised when an engine's weights/dependencies are not installed yet."""


class EngineNotFoundError(OneManDubbingError):
    """Raised when a requested engine name is not present in the registry."""


class TrainingFailedError(OneManDubbingError):
    """Raised when a training run fails (e.g. GPU out of memory, bad data)."""


class TrainingInProgressError(OneManDubbingError):
    """Raised when a new training run is requested while one is already
    active. Training is GPU/CPU-exclusive - running two at once starves
    both and can make the whole machine unresponsive.
    """


class ConversionFailedError(OneManDubbingError):
    """Raised when voice conversion / inference fails."""


class SynthesisFailedError(OneManDubbingError):
    """Raised when zero-shot voice synthesis fails."""


class AudioValidationError(OneManDubbingError):
    """Raised when an audio file is missing, corrupt, or unsupported."""


class DownloadError(OneManDubbingError):
    """Raised when a model/weights download fails or is incomplete."""


class InstallationError(OneManDubbingError):
    """Raised when a setup step (GPU detection, torch install, etc) fails."""
