from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from core.domain.errors import OneManDubbingError


class ProcessRunError(OneManDubbingError):
    """Raised when a subprocess exits with a non-zero status."""


def run_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command to completion and return its combined stdout+stderr.

    This is the single allowed call site for subprocess.run in the project;
    everything else must go through here (DRY, and a single place to strip
    third-party stack traces before they reach the UI).
    """
    result = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProcessRunError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stderr.strip()}"
        )
    return result.stdout


def stream_command(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Run a command, yielding stdout line by line as it becomes available.

    Used for long-running steps (training, model downloads) whose progress
    must be forwarded to the UI while still running.
    """
    process = subprocess.Popen(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip("\n")
        if on_line is not None:
            on_line(stripped)
        yield stripped

    return_code = process.wait()
    if return_code != 0:
        raise ProcessRunError(f"Command failed ({return_code}): {' '.join(args)}")
