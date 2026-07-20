# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json

import pytest

from features.comfy.coordinator import (
    AcceleratorBusyError,
    AcceleratorCoordinationMiddleware,
    AcceleratorCoordinator,
)


def test_accelerator_slot_rejects_cross_tab_interleaving():
    async def scenario():
        coordinator = AcceleratorCoordinator()
        async with coordinator.slot("a chat response"):
            assert coordinator.busy is True
            assert coordinator.operation == "a chat response"
            with pytest.raises(AcceleratorBusyError, match = "image generation"):
                async with coordinator.slot("image generation"):
                    pass
        assert coordinator.busy is False

    asyncio.run(scenario())


def test_middleware_holds_slot_until_stream_response_finishes(monkeypatch):
    async def scenario():
        import features.comfy.coordinator as module

        coordinator = AcceleratorCoordinator()
        monkeypatch.setattr(module, "accelerator_coordinator", coordinator)
        stream_started = asyncio.Event()
        finish_stream = asyncio.Event()

        async def streaming_app(_scope, _receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            stream_started.set()
            await finish_stream.wait()
            await send({"type": "http.response.body", "body": b"done"})

        async def _true():
            return True

        middleware = AcceleratorCoordinationMiddleware(streaming_app)
        monkeypatch.setattr(middleware, "_authenticated", lambda _scope: _true())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/inference/chat/completions",
        }
        first_messages = []

        async def first_send(message):
            first_messages.append(message)

        first = asyncio.create_task(middleware(scope, lambda: None, first_send))
        await stream_started.wait()

        second_messages = []

        async def second_send(message):
            second_messages.append(message)

        await middleware(scope, lambda: None, second_send)
        assert second_messages[0]["status"] == 409
        detail = json.loads(second_messages[1]["body"])["detail"]
        assert "chat response" in detail

        finish_stream.set()
        await first
        assert first_messages[-1]["body"] == b"done"
        assert coordinator.busy is False

    asyncio.run(scenario())
