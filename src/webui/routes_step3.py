from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse

from config import DEFAULT_OUTPUT_FORMAT
from core.domain.errors import OneManDubbingError
from core.services.dubbing_service import DubbingService
from core.services.training_service import TrainingService
from infra.filesystem_paths import output_path_for
from webui.template_engine import templates

router = APIRouter(prefix="/step3", tags=["step3"])
_service = DubbingService()
_training_service = TrainingService()


@router.get("/")
async def show_step3(request: Request, voice_name: str):
    return templates.TemplateResponse(
        request,
        "step3_dubbing.html",
        {"voice_name": voice_name, "models": _training_service.list_models(voice_name)},
    )


@router.post("/dub")
async def dub_audio(
    request: Request,
    voice_name: str = Form(...),
    model_id: str = Form(...),
    actor_audio: UploadFile = ...,
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    model = _training_service.get_model(voice_name, model_id)
    if model is None:
        return await _show_error(request, voice_name, f"Trained model not found: {model_id}")

    job_id = uuid.uuid4().hex
    raw_bytes = await actor_audio.read()

    try:
        output_asset = await asyncio.to_thread(
            _service.dub,
            voice_name,
            model_id,
            raw_bytes,
            job_id,
            model.engine_name,
            output_format,
        )
    except OneManDubbingError as exc:
        return await _show_error(request, voice_name, str(exc))

    return templates.TemplateResponse(
        request,
        "step3_dubbing.html",
        {
            "voice_name": voice_name,
            "models": _training_service.list_models(voice_name),
            "output_path": str(output_asset.path.name),
            "job_id": job_id,
        },
    )


@router.get("/download/{voice_name}/{job_id}")
async def download_output(voice_name: str, job_id: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
    file_path = output_path_for(voice_name, job_id, output_format)
    return FileResponse(path=file_path, filename=file_path.name)


async def _show_error(request: Request, voice_name: str, message: str):
    return templates.TemplateResponse(
        request,
        "step3_dubbing.html",
        {
            "voice_name": voice_name,
            "models": _training_service.list_models(voice_name),
            "error": message,
        },
        status_code=400,
    )
