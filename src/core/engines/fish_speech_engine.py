from __future__ import annotations

import sys
import uuid

from core.domain.audio_asset import AudioAsset
from core.domain.errors import SynthesisFailedError
from core.domain.voice_profile import SynthesisParameters
from infra.audio_io import load_audio_asset
from infra.filesystem_paths import FISH_SPEECH_DIR, REFERENCES_DIR
from infra.process_runner import ProcessRunError, run_command


class FishSpeechEngine:
    """VoiceSynthesisEngine implementation backed by the vendored Fish Speech repo.

    Unlike RvcEngine, this integration is unverified against the real
    fish-speech CLI (see README "Notes on the vendored engines") — the
    script path and flags below are the intended integration point, not
    confirmed against the actual upstream repo.
    """

    engine_name = "fish_speech"

    def is_ready(self) -> bool:
        checkpoint = FISH_SPEECH_DIR / "weights" / "model.pth"
        return FISH_SPEECH_DIR.is_dir() and checkpoint.is_file()

    def synthesize(self, text_sample: str, parameters: SynthesisParameters) -> AudioAsset:
        output_path = REFERENCES_DIR / f"synthesized_{uuid.uuid4().hex}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        synthesis_script = FISH_SPEECH_DIR / "tools" / "synthesize_cli.py"
        args = [
            sys.executable,
            str(synthesis_script),
            "--text",
            text_sample,
            "--pitch",
            str(parameters.pitch_base),
            "--speed",
            str(parameters.speed),
            "--output",
            str(output_path),
        ]
        if parameters.timbre_embedding:
            args += ["--timbre", parameters.timbre_embedding]
        if parameters.accent:
            args += ["--accent", parameters.accent]

        try:
            run_command(args, cwd=FISH_SPEECH_DIR)
        except ProcessRunError as exc:
            raise SynthesisFailedError(f"Fish Speech synthesis failed: {exc}") from exc

        if not output_path.is_file():
            raise SynthesisFailedError(f"Fish Speech did not produce an output file: {output_path}")
        return load_audio_asset(output_path)
