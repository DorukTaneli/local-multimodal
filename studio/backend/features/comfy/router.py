# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated Studio API for local ComfyUI generation and asset serving."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import secrets
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from auth.authentication import get_current_subject
from .assets import AssetError, resolve_png_asset, save_png
from .client import (
    ComfyCheckpointError,
    ComfyClient,
    ComfyError,
    ComfyGenerationTimeout,
    ComfyUnavailableError,
)
from .models import (
    GenerateRequest,
    GeneratedAsset,
    GenerateResponse,
    JSON_SAFE_INTEGER_MAX,
    ReleaseResponse,
    StatusResponse,
)
from .workflow import (
    WorkflowError,
    checkpoint_name,
    effective_prompt,
    image_dimensions,
    load_workflow,
    mutate_workflow,
)


router = APIRouter(dependencies = [Depends(get_current_subject)])
_operation_lock = asyncio.Lock()


class ComfyBusyError(RuntimeError):
    pass


@asynccontextmanager
async def operation_slot():
    # Lock.acquire() does not suspend when the lock is free, so this check and
    # acquisition are atomic with respect to other tasks on the same event loop.
    if _operation_lock.locked():
        raise ComfyBusyError("ComfyUI is busy with image generation or memory release.")
    await _operation_lock.acquire()
    try:
        yield
    finally:
        _operation_lock.release()


def _request_prompt(payload: GenerateRequest, workflow: dict) -> str:
    reroll = payload.effectivePrompt is not None
    initial_supplied = payload.imageTags is not None or payload.conversationTags is not None
    if reroll and initial_supplied:
        raise ValueError("Provide either effectivePrompt or imageTags and conversationTags, not both.")
    if reroll:
        if not payload.effectivePrompt or not payload.effectivePrompt.strip():
            raise ValueError("effectivePrompt cannot be blank.")
        return payload.effectivePrompt
    if payload.imageTags is None or payload.conversationTags is None:
        raise ValueError("imageTags and conversationTags are required for initial generation.")
    return effective_prompt(
        workflow,
        image_tags = payload.imageTags,
        conversation_tags = payload.conversationTags,
    )


@router.get("/status", response_model = StatusResponse)
async def status() -> StatusResponse:
    failures: list[str] = []
    workflow = None
    checkpoint = None
    try:
        workflow = load_workflow()
        checkpoint = checkpoint_name(workflow)
        workflow_valid = True
    except WorkflowError as exc:
        workflow_valid = False
        failures.append(str(exc))

    reachable = False
    checkpoint_available = False
    try:
        async with ComfyClient() as client:
            await client.check_reachable()
            reachable = True
            if checkpoint:
                checkpoints = await client.available_checkpoints()
                checkpoint_available = checkpoint in checkpoints
                if not checkpoint_available:
                    failures.append(
                        f"ComfyUI checkpoint {checkpoint!r} is not installed. "
                        "Add it to ComfyUI's checkpoints folder and refresh ComfyUI."
                    )
    except ComfyError as exc:
        failures.append(str(exc))

    busy = _operation_lock.locked()
    if busy:
        failures.append("ComfyUI is busy with image generation or memory release.")
    ready = reachable and workflow_valid and checkpoint_available and not busy
    return StatusResponse(
        ready = ready,
        reachable = reachable,
        workflowValid = workflow_valid,
        checkpointAvailable = checkpoint_available,
        busy = busy,
        checkpoint = checkpoint,
        failures = failures,
    )


@router.post("/generate", response_model = GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    try:
        async with operation_slot():
            workflow = load_workflow()
            try:
                prompt = _request_prompt(payload, workflow)
            except ValueError as exc:
                raise HTTPException(status_code = 400, detail = str(exc)) from exc
            checkpoint = checkpoint_name(workflow)
            # JSON numbers are decoded as IEEE-754 doubles by the frontend.
            # Stay within the exact integer range so persisted seeds remain reproducible.
            seed = secrets.randbelow(JSON_SAFE_INTEGER_MAX + 1)
            generated_workflow = mutate_workflow(
                workflow,
                prompt = prompt,
                seed = seed,
                filename_prefix = f"LocalMultimodal_{uuid4().hex}",
            )
            width, height = image_dimensions(workflow)
            async with ComfyClient() as client:
                await client.preflight(checkpoint)
                _prompt_id, png = await client.generate(generated_workflow, save_node_id = "7")
            asset_id = save_png(png)
            return GenerateResponse(
                effectivePrompt = prompt,
                asset = GeneratedAsset(
                    assetId = str(asset_id),
                    seed = seed,
                    width = width,
                    height = height,
                ),
            )
    except ComfyBusyError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except WorkflowError as exc:
        raise HTTPException(status_code = 503, detail = str(exc)) from exc
    except ComfyCheckpointError as exc:
        raise HTTPException(status_code = 503, detail = str(exc)) from exc
    except ComfyUnavailableError as exc:
        raise HTTPException(status_code = 503, detail = str(exc)) from exc
    except ComfyGenerationTimeout as exc:
        raise HTTPException(status_code = 504, detail = str(exc)) from exc
    except (ComfyError, AssetError) as exc:
        raise HTTPException(status_code = 502, detail = str(exc)) from exc


@router.post("/release", response_model = ReleaseResponse)
async def release() -> ReleaseResponse:
    try:
        async with operation_slot():
            try:
                async with ComfyClient() as client:
                    await client.release()
            except ComfyUnavailableError:
                return ReleaseResponse(offline = True)
            except ComfyError as exc:
                raise HTTPException(status_code = 502, detail = str(exc)) from exc
            return ReleaseResponse()
    except ComfyBusyError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.get("/assets/{asset_id}")
async def asset(asset_id: str) -> FileResponse:
    try:
        path = resolve_png_asset(asset_id)
    except AssetError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc
    return FileResponse(path, media_type = "image/png")
