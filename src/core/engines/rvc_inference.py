from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from core.domain.dubbing_job import DubbingJob, DubbingStatus
from core.domain.errors import ConversionFailedError
from infra.filesystem_paths import RVC_ASSETS_DIR, RVC_DIR
from infra.job_progress_bus import progress_bus
from infra.process_runner import ProcessRunError, stream_command

OnLine = Callable[[str], None]


def _has_similarity_index(voice_name: str) -> bool:
    indices_dir = RVC_ASSETS_DIR / "indices"
    return indices_dir.is_dir() and any(indices_dir.glob(f"{voice_name}_*"))


def run_inference(
    source_path: Path, model_path: Path, output_path: Path, on_line: OnLine | None = None
) -> Path:
    """Invoke the vendored RVC inference CLI (infer/cli.py) to convert one
    audio file.

    Kept separate from rvc_engine.py so that engine orchestration (job
    bookkeeping, progress reporting) and the raw subprocess call to the
    vendored RVC scripts don't live in the same 135-line file.

    Run as "-m infer.cli" rather than by file path: cwd is RVC_DIR, and -m
    puts that on sys.path[0], which is what lets cli.py's own top-level
    imports (configs, i18n, tools) resolve.
    """
    if not model_path.exists():
        raise ConversionFailedError(f"Trained model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Resolve to absolute paths: the subprocess runs with cwd=RVC_DIR, so a
    # relative path here would be (mis)resolved against that instead of the
    # caller's own working directory.
    args = [
        sys.executable,
        "-m",
        "infer.cli",
        "--input",
        str(source_path.resolve()),
        "--model",
        str(model_path.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    if not _has_similarity_index(model_path.stem):
        # No index was built for this voice (use_similarity_index was off
        # during training) - infer/cli.py otherwise refuses to run.
        args += ["--index-rate", "0"]

    try:
        for line in stream_command(args, cwd=RVC_DIR):
            if on_line is not None:
                on_line(line)
    except ProcessRunError as exc:
        raise ConversionFailedError(f"RVC inference failed: {exc}") from exc

    if not output_path.is_file():
        raise ConversionFailedError(f"RVC inference did not produce an output file: {output_path}")
    return output_path


def run_inference_with_progress(
    source_path: Path, model_path: Path, output_path: Path, job_id: str
) -> Path:
    """run_inference(), publishing a DubbingJob's progress/log lines as it
    runs. Does not close the job on success - the caller (DubbingService)
    still has a final output-format conversion step to do and owns marking
    the job truly finished.
    """
    job = DubbingJob(job_id=job_id, model_id=model_path.stem)
    progress_bus.publish(job)

    def on_line(line: str) -> None:
        job.append_log_line(line)
        progress_bus.publish(job)

    try:
        result_path = run_inference(source_path, model_path, output_path, on_line)
    except ConversionFailedError as exc:
        job.status = DubbingStatus.FAILED
        job.error_message = str(exc)
        progress_bus.publish(job)
        progress_bus.close(job_id)
        raise

    job.output_path = result_path
    progress_bus.publish(job)
    return result_path
