# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic accelerator handoffs between Studio chat and local ComfyUI."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .client import (
    ComfyClient,
    ComfyError,
    ComfyReleaseTimeout,
    ComfyReleaseUnverifiableError,
    ComfyUnavailableError,
)


class AcceleratorBusyError(RuntimeError):
    pass


class ChatModelUnloadError(RuntimeError):
    pass


class AcceleratorCoordinator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._operation: str | None = None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def operation(self) -> str | None:
        return self._operation

    @asynccontextmanager
    async def slot(self, operation: str):
        # asyncio.Lock.acquire() cannot suspend while unlocked, so this remains
        # atomic for requests handled by the process event loop.
        if self._lock.locked():
            active = self._operation or "another accelerator operation"
            raise AcceleratorBusyError(f"Cannot start {operation} while {active} is active.")
        await self._lock.acquire()
        self._operation = operation
        try:
            yield
        finally:
            self._operation = None
            self._lock.release()

    async def prepare_chat_model_load(self) -> None:
        """Release Comfy before upstream inference begins a local model load."""
        try:
            async with ComfyClient() as client:
                await client.release()
        except ComfyUnavailableError:
            # An offline ComfyUI process owns no accelerator allocation.
            return

    async def unload_chat_model(self) -> None:
        """Unload either local inference backend while the accelerator slot is held."""
        from core.inference.llama_keepwarm import inference_lifecycle_gate, note_model_unloaded
        from routes.inference import get_inference_backend, get_llama_cpp_backend

        try:
            async with inference_lifecycle_gate():
                unloaded = False
                llama_backend = get_llama_cpp_backend()
                if llama_backend.is_active:
                    await asyncio.to_thread(llama_backend.unload_model)
                    unloaded = True

                backend = get_inference_backend()
                active_model = getattr(backend, "active_model_name", None)
                if active_model:
                    await asyncio.to_thread(backend.unload_model, active_model)
                    unloaded = True

                if unloaded:
                    note_model_unloaded()
        except Exception as exc:
            raise ChatModelUnloadError(
                "The local chat model could not be unloaded for image generation."
            ) from exc


accelerator_coordinator = AcceleratorCoordinator()


class AcceleratorCoordinationMiddleware:
    """Hold the coordinator through local chat streams and model loads."""

    _CHAT_PATHS = {
        "/api/inference/chat/completions",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/responses",
    }

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    async def _authenticated(scope: Scope) -> bool:
        # Middleware runs before FastAPI dependencies. Validate first so an
        # unauthenticated /load request cannot trigger a Comfy memory release.
        from auth.authentication import get_current_subject

        headers = dict(scope.get("headers") or [])
        raw = headers.get(b"authorization", b"").decode("latin-1")
        scheme, separator, token = raw.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token:
            return False
        try:
            await get_current_subject(
                HTTPAuthorizationCredentials(scheme = scheme, credentials = token)
            )
        except HTTPException:
            return False
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        operation = None
        if path == "/api/inference/load":
            operation = "chat model loading"
        elif path in self._CHAT_PATHS:
            operation = "a chat response"
        if operation is None:
            await self.app(scope, receive, send)
            return
        if not await self._authenticated(scope):
            await self.app(scope, receive, send)
            return

        try:
            async with accelerator_coordinator.slot(operation):
                if path == "/api/inference/load":
                    await accelerator_coordinator.prepare_chat_model_load()
                await self.app(scope, receive, send)
        except AcceleratorBusyError as exc:
            await JSONResponse({"detail": str(exc)}, status_code = 409)(scope, receive, send)
        except ComfyReleaseUnverifiableError as exc:
            await JSONResponse({"detail": str(exc)}, status_code = 409)(scope, receive, send)
        except ComfyReleaseTimeout as exc:
            await JSONResponse({"detail": str(exc)}, status_code = 504)(scope, receive, send)
        except ComfyError as exc:
            await JSONResponse({"detail": str(exc)}, status_code = 502)(scope, receive, send)
