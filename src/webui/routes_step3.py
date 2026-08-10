from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from config import DEFAULT_OUTPUT_FORMAT
from core.domain.dubbing_job import DubbingStatus
from core.domain.errors import DubbingInProgressError
from core.services.dubbing_service import DubbingService
from core.services.reference_voice_service import ReferenceVoiceService
from core.services.training_service import TrainingService
from infra.filesystem_paths import output_path_for
from infra.job_progress_bus import progress_bus
from webui.template_engine import templates

router = APIRouter(prefix="/step3", tags=["step3"])
_service = DubbingService()
_training_service = TrainingService()
_voice_service = ReferenceVoiceService()


@router.get("/")
async def show_step3(request: Request, voice_name: str | None = None):
    models = _training_service.list_models(voice_name) if voice_name else []
    return templates.TemplateResponse(
        request,
        "step3_dubbing.html",
        {"voices": _voices_with_models(), "voice_name": voice_name, "models": models},
    )


@router.post("/dub")
async def dub_audio(
    voice_name: str = Form(...),
    model_id: str = Form(...),
    actor_audio: UploadFile = ...,
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    model = _training_service.get_model(voice_name, model_id)
    if model is None:
        return JSONResponse({"error": f"Trained model not found: {model_id}"}, status_code=404)

    raw_bytes = await actor_audio.read()
    try:
        job_id = _service.start_dub_in_background(
            voice_name, model_id, raw_bytes, model.engine_name, output_format
        )
    except DubbingInProgressError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return {"status": "started", "voice_name": voice_name, "job_id": job_id, "output_format": output_format}


@router.get("/progress/{job_id}")
async def stream_progress(job_id: str, voice_name: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
    async def event_stream():
        while True:
            job = progress_bus.latest_for_job(job_id)
            if job is not None:
                payload = {
                    "status": job.status.value,
                    "error_message": job.error_message,
                    "log_lines": job.recent_log_lines,
                    "job_id": job.job_id,
                    "download_url": f"/step3/download/{voice_name}/{job_id}?output_format={output_format}"
                    if job.status == DubbingStatus.COMPLETED
                    else None,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if job.status in (DubbingStatus.COMPLETED, DubbingStatus.FAILED):
                    break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/download/{voice_name}/{job_id}")
async def download_output(voice_name: str, job_id: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
    file_path = output_path_for(voice_name, job_id, output_format)
    return FileResponse(path=file_path, filename=file_path.name)


def _voices_with_models() -> list[dict]:
    result = []
    for voice in _voice_service.list_voices():
        model_count = len(_training_service.list_models(voice.name))
        if model_count > 0:
            result.append({"name": voice.name, "model_count": model_count})
    return result
