from __future__ import annotations

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

from core.domain.errors import OneManDubbingError
from core.domain.voice_profile import SynthesisParameters
from core.services.reference_voice_service import ReferenceVoiceService
from webui.template_engine import templates

router = APIRouter(prefix="/step1", tags=["step1"])
_service = ReferenceVoiceService()


@router.get("/")
async def show_step1(request: Request):
    return templates.TemplateResponse(
        request,
        "step1_reference.html",
        {"engines": _service.list_available_synthesis_engines()},
    )


@router.post("/upload")
async def upload_reference(
    request: Request, voice_name: str = Form(...), audio_file: UploadFile = ...
):
    try:
        raw_bytes = await audio_file.read()
        _service.create_from_upload(voice_name, raw_bytes)
    except OneManDubbingError as exc:
        return templates.TemplateResponse(
            request, "step1_reference.html", {"error": str(exc)}, status_code=400
        )
    return RedirectResponse(url=f"/step2/?voice_name={voice_name}", status_code=303)


@router.post("/synthesize")
async def synthesize_reference(
    request: Request,
    voice_name: str = Form(...),
    text_sample: str = Form(...),
    engine_name: str = Form(...),
    pitch_base: float = Form(0.0),
    speed: float = Form(1.0),
    accent: str | None = Form(None),
):
    parameters = SynthesisParameters(pitch_base=pitch_base, speed=speed, accent=accent)
    try:
        _service.create_from_synthesis(voice_name, text_sample, parameters, engine_name)
    except OneManDubbingError as exc:
        return templates.TemplateResponse(
            request, "step1_reference.html", {"error": str(exc)}, status_code=400
        )
    return RedirectResponse(url=f"/step2/?voice_name={voice_name}", status_code=303)
