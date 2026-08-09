# OneManDubbing

A 100% local pipeline that lets a single voice actor record lines in their own
voice and convert them into a distinct target character voice, for use in
short-film production and amateur dubbing. Runs entirely on the user's GPU
(reference target: RTX 3060, 12 GB).

## Three-step pipeline

1. **Target voice definition** — either upload a reference audio (Route A) or
   synthesize one from scratch with Fish Speech (Route B). Both routes
   converge on a single `ReferenceVoiceSource` / `VoiceProfile` artifact.
2. **Training** — an RVC model is trained locally from that reference audio.
3. **Dubbing (inference)** — the actor uploads their own recording; the
   trained model converts it into the target voice as an `.mp3`/`.wav`.

An automatic installer resolves every dependency (CUDA-matched PyTorch,
engine repos, base model weights) with no manual steps beyond having the
NVIDIA driver already installed.

## Architecture principles

- **Extreme modularity** — no Python file exceeds 135 lines; every module has
  a single responsibility.
- **OOP + strong typing** — full type hints (including return types) on every
  public interface; dataclasses for data models; `Protocol`s for contracts
  between layers.
- **Engine-agnostic** — voice conversion (RVC) and zero-shot synthesis (Fish
  Speech) are plugged in through `VoiceConversionEngine` and
  `VoiceSynthesisEngine` Protocols, discovered at runtime via
  `EngineRegistry`. Adding a new engine means implementing the Protocol and
  registering it — no changes to the UI, orchestration, or installer.
- **DRY** — subprocess calls, HTTP/Hugging Face downloads, audio I/O and
  progress reporting each live in exactly one module under `infra/`.
- **Strict layering** — `core/domain` never imports `core/engines`; `webui`
  only talks to `core/services`, never to engines or infra directly.

## Getting started

```bash
make install   # or: pip install -r requirements.txt
make run        # or: cd src && python main.py
```

On first run this automatically triggers the full setup sequence (GPU
detection → PyTorch install → clone RVC/Fish Speech into `vendor/` and
download their weights → project dependencies → validate). Subsequent runs
skip straight to starting the web UI at `http://127.0.0.1:7860`.

To run setup explicitly:

```bash
make setup     # or: cd src && python -m installer.run_installer
```

See the [Makefile](Makefile) (`make help`) for the full list of commands,
including `make dev` for an auto-reloading server during development.

## Project layout

```
.
├── requirements.txt   Python dependencies
├── Makefile            install / setup / run / dev / clean
├── src/
│   ├── installer/     one-time automatic setup (GPU, torch, deps, engines, weights)
│   ├── core/
│   │   ├── domain/    dataclasses and domain errors — no external dependencies
│   │   ├── engines/   VoiceSynthesisEngine / VoiceConversionEngine Protocols + RVC/Fish Speech implementations
│   │   ├── registry/  EngineRegistry — runtime engine discovery
│   │   └── services/  orchestration for steps 1, 2 and 3
│   ├── infra/         subprocess wrapper, Hugging Face client, audio I/O, progress bus, paths
│   ├── webui/         FastAPI app, routes, Jinja2 templates, brutalist CSS/JS
│   ├── config.py      declarative EngineManifest table and project defaults
│   └── main.py        entry point — runs first-time setup if needed, then serves the UI
├── vendor/             RVC and Fish Speech repos, cloned by the installer (gitignored)
└── data/               references/, models/, outputs/ generated at runtime (gitignored)
```

`vendor/` and `data/` live at the repo root, outside `src/`, and are
gitignored as whole directories. Neither is source: `vendor/` is installer
output (multi-GB cloned repos plus their downloaded weights — regenerated
by `make setup`), and `data/` is content the user generates by using the
app (their own recordings, dubbed outputs). Keeping them out of `src/`
means `src/` stays 100% source code.

Downloaded weights live *inside* `vendor/<engine>/` (e.g.
`vendor/rvc/assets/hubert_base/`), not in a separate shared folder: real
engine code hardcodes weight paths relative to its own repo root (see
`infer/hubert.py`, `infer/cli.py` in the vendored RVC repo), so that's
where they have to be for the engine to find them.

## Notes on the vendored engines

**RVC** (`core/engines/rvc_engine.py`, `rvc_preprocessing.py`,
`rvc_manifest_builder.py`, `rvc_layout.py`, `rvc_inference.py`) is a
verified, working integration against the real
`RVC-Project/Retrieval-based-Voice-Conversion-WebUI` repo. Real RVC training
is a 5-stage pipeline, not a single command:

1. `train.preprocess` — slice the reference audio into training segments
2. `train.dataset.extract_f0` — pitch extraction (RMVPE, CUDA)
3. `train.dataset.extract_hubert_feature` — HuBERT feature extraction
4. build `filelist.txt` + `config.json` for the run (`rvc_manifest_builder.py`)
5. `train.train` — the actual GAN training loop, then optionally
   `train.train_index` to build a similarity index

Everything is invoked as `python -m train.preprocess` (etc.), never by file
path: `vendor/rvc/train/train.py` is a module *inside* the `train` package
that shares the package's own name, which shadows the real `train` package
on `sys.path` if invoked by path instead of `-m`.

**Fish Speech** (`fish_speech_engine.py`) is **not** verified — its script
path (`tools/synthesize_cli.py`) and weight layout are the intended
integration point, not confirmed against the actual upstream repo. Treat
Route B (synthesize-from-scratch) as unimplemented until this is checked
the same way RVC was.

**Shared Python environment risk**: `vendor/rvc`'s own dependency file
pulls in `gradio` (for its own bundled webui.py, which this project doesn't
use), which pins an old `fastapi`/`starlette`. Since engine dependencies and
this project's dependencies install into the same environment,
`run_installer.py` deliberately installs engine requirements *before* this
project's own `requirements.txt`, so our pinned versions win last. If you
manually reinstall an engine's requirements afterward, re-run `make install`
to restore ours. The robust fix would be an isolated virtualenv per engine;
that hasn't been done.

## Conventions

- Every public function/method is fully typed, including its return type;
  every module starts with `from __future__ import annotations`.
- Data models are `@dataclass(frozen=True)` whenever immutability makes sense.
- No `subprocess.run`, HTTP call, or direct file I/O outside `infra/`.
- Engine failures (training failure, out-of-memory, incomplete download) are
  raised as domain exceptions from `core/domain/errors.py` — never as raw
  third-party stack traces reaching the UI.
- Each file stays at or under 135 lines; a module that grows past that is a
  signal to split it (e.g. separate validation from orchestration).
