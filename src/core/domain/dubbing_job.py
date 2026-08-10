from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DubbingStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DubbingJob:
    """Mutable state/progress of a single Step 3 conversion run.

    Mirrors TrainingJob's shape (see core/domain/training_job.py) so both
    can share the same progress bus and the same SSE/log-panel pattern on
    the frontend.
    """

    job_id: str
    model_id: str
    status: DubbingStatus = DubbingStatus.RUNNING
    output_path: Path | None = None
    error_message: str | None = None
    recent_log_lines: list[str] = field(default_factory=list)

    _MAX_LOG_LINES = 20

    def append_log_line(self, line: str) -> None:
        if not line.strip():
            return
        self.recent_log_lines.append(line)
        del self.recent_log_lines[: -self._MAX_LOG_LINES]
