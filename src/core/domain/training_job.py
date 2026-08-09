from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingJob:
    """Mutable state/progress of a single training run.

    Deliberately not frozen: the training service updates it in place as the
    engine process reports progress, and the same instance is what gets
    published on the progress bus.
    """

    voice_name: str
    engine_name: str
    job_id: str = field(default_factory=lambda: str(uuid4()))
    status: TrainingStatus = TrainingStatus.QUEUED
    current_epoch: int = 0
    total_epochs: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_output_path: Path | None = None
    error_message: str | None = None
    recent_log_lines: list[str] = field(default_factory=list)

    _MAX_LOG_LINES = 20

    @property
    def progress_ratio(self) -> float:
        if self.total_epochs <= 0:
            return 0.0
        return min(self.current_epoch / self.total_epochs, 1.0)

    def append_log_line(self, line: str) -> None:
        if not line.strip():
            return
        self.recent_log_lines.append(line)
        del self.recent_log_lines[: -self._MAX_LOG_LINES]
