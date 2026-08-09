from __future__ import annotations

from pathlib import Path

from core.domain.errors import ConversionFailedError
from infra.filesystem_paths import RVC_DIR
from infra.process_runner import ProcessRunError, run_command


def run_inference(source_path: Path, model_path: Path, output_path: Path) -> Path:
    """Invoke the vendored RVC inference CLI to convert one audio file.

    Kept separate from rvc_engine.py so that engine orchestration (job
    bookkeeping, progress reporting) and the raw subprocess call to the
    vendored RVC scripts don't live in the same 135-line file.
    """
    if not model_path.exists():
        raise ConversionFailedError(f"Trained model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference_script = RVC_DIR / "tools" / "infer_cli.py"

    try:
        run_command(
            [
                "python",
                str(inference_script),
                "--input",
                str(source_path),
                "--model",
                str(model_path),
                "--output",
                str(output_path),
            ],
            cwd=RVC_DIR,
        )
    except ProcessRunError as exc:
        raise ConversionFailedError(f"RVC inference failed: {exc}") from exc

    if not output_path.is_file():
        raise ConversionFailedError(f"RVC inference did not produce an output file: {output_path}")
    return output_path
