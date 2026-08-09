from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse

from config import DEFAULT_CONVERSION_ENGINE, DEFAULT_OUTPUT_FORMAT
from core.domain.errors import OneManDubbingError
from core.services.dubbing_service import DubbingService
from infra.filesystem_paths import output_path_for
from webui.template_engine import templates

router = APIRouter(prefix="/step3", tags=["step3"])
_service = DubbingService()


@router.get("/")
async def show_step3(request: Request, voice_name: str):
    return templates.TemplateResponse(
        request, "step3_dubbing.html", {"voice_name": voice_name}
    )


@router.post("/dub")
async def dub_audio(
    request: Request,
    voice_name: str = Form(...),
    actor_audio: UploadFile = ...,
    engine_name: str = Form(DEFAULT_CONVERSION_ENGINE),
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    job_id = uuid.uuid4().hex
    raw_bytes = await actor_audio.read()

    try:
        output_asset = await asyncio.to_thread(
            _service.dub, voice_name, raw_bytes, job_id, engine_name, output_format
        )
    except OneManDubbingError as exc:
        return templates.TemplateResponse(
            request, "step3_dubbing.html", {"voice_name": voice_name, "error": str(exc)}, status_code=400
        )

    return templates.TemplateResponse(
        request,
        "step3_dubbing.html",
        {"voice_name": voice_name, "output_path": str(output_asset.path.name), "job_id": job_id},
    )


@router.get("/download/{voice_name}/{job_id}")
async def download_output(voice_name: str, job_id: str, output_format: str = DEFAULT_OUTPUT_FORMAT):
    file_path = output_path_for(voice_name, job_id, output_format)
    return FileResponse(path=file_path, filename=file_path.name)
