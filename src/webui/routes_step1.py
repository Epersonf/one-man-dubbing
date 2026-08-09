from __future__ import annotations

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

from core.domain.errors import OneManDubbingError
from core.domain.voice_profile import SynthesisParameters
from core.services.reference_voice_service import ReferenceVoiceService
from core.services.training_service import TrainingService
from webui.template_engine import templates

router = APIRouter(prefix="/step1", tags=["step1"])
_service = ReferenceVoiceService()
_training_service = TrainingService()


@router.get("/")
async def show_step1(request: Request):
    return templates.TemplateResponse(
        request,
        "step1_reference.html",
        {
            "engines": _service.list_available_synthesis_engines(),
            "voices": _service.list_voices(),
        },
    )


@router.post("/upload")
async def upload_reference(
    request: Request, voice_name: str = Form(...), audio_file: UploadFile = ...
):
    try:
        raw_bytes = await audio_file.read()
        _service.create_from_upload(voice_name, raw_bytes)
    except OneManDubbingError as exc:
        return await _show_error(request, exc)
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
        return await _show_error(request, exc)
    return RedirectResponse(url=f"/step2/?voice_name={voice_name}", status_code=303)


@router.post("/delete")
async def delete_voice(voice_name: str = Form(...)):
    for model in _training_service.list_models(voice_name):
        _training_service.delete_model(voice_name, model.model_id, model.engine_name)
    _service.delete_voice(voice_name)
    return RedirectResponse(url="/step1/", status_code=303)


async def _show_error(request: Request, exc: Exception):
    return templates.TemplateResponse(
        request,
        "step1_reference.html",
        {
            "error": str(exc),
            "engines": _service.list_available_synthesis_engines(),
            "voices": _service.list_voices(),
        },
        status_code=400,
    )
