from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import DEFAULT_CONVERSION_ENGINE
from core.domain.training_config import TrainingConfig
from core.domain.training_job import TrainingStatus
from core.services.reference_voice_service import ReferenceVoiceService
from core.services.training_service import TrainingService
from infra.job_progress_bus import progress_bus
from webui.template_engine import templates

router = APIRouter(prefix="/step2", tags=["step2"])
_service = TrainingService()
_voice_service = ReferenceVoiceService()


@router.get("/")
async def show_step2(request: Request, voice_name: str):
    return templates.TemplateResponse(
        request,
        "step2_training.html",
        {
            "voice_name": voice_name,
            "engines": _service.list_available_engines(),
            "models": _service.list_models(voice_name),
        },
    )


@router.post("/train")
async def start_training(
    voice_name: str = Form(...),
    engine_name: str = Form(DEFAULT_CONVERSION_ENGINE),
    epochs: int = Form(200),
    sample_rate: int = Form(40_000),
    use_similarity_index: bool = Form(True),
):
    voice_profile = _voice_service.get_voice(voice_name)
    if voice_profile is None:
        return JSONResponse({"error": f"Voice not found: {voice_name}"}, status_code=404)

    config = TrainingConfig(
        epochs=epochs, sample_rate=sample_rate, use_similarity_index=use_similarity_index
    )
    model_id = _service.start_training_in_background(voice_profile, config, engine_name)
    return {"status": "started", "voice_name": voice_name, "model_id": model_id}


@router.post("/delete-model")
async def delete_model(voice_name: str = Form(...), model_id: str = Form(...), engine_name: str = Form(...)):
    _service.delete_model(voice_name, model_id, engine_name)
    return {"status": "deleted", "model_id": model_id}


@router.get("/progress/{model_id}")
async def stream_progress(model_id: str):
    async def event_stream():
        while True:
            job = progress_bus.latest_for_job(model_id)
            if job is not None:
                payload = {
                    "status": job.status.value,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.total_epochs,
                    "progress_ratio": job.progress_ratio,
                    "error_message": job.error_message,
                    "model_id": job.job_id,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
                    break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
