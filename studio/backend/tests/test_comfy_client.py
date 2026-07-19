# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from features.comfy.client import (
    ComfyCheckpointError,
    ComfyClient,
    ComfyGenerationError,
    ComfyGenerationTimeout,
    ComfyReleaseTimeout,
    ComfyUnavailableError,
)


PNG = b"\x89PNG\r\n\x1a\nimage-data"


def _drive(coro):
    return asyncio.run(coro)


def _client(handler, **kwargs) -> tuple[ComfyClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(
        base_url = "http://comfy.test", transport = httpx.MockTransport(handler)
    )
    return ComfyClient(client = http, base_url = "http://comfy.test", **kwargs), http


def test_preflight_reports_missing_checkpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json = {})
        return httpx.Response(200, json = ["another.safetensors"])

    client, http = _client(handler)
    with pytest.raises(ComfyCheckpointError, match = "not installed"):
        _drive(client.preflight("required.safetensors"))
    _drive(http.aclose())


def test_unreachable_comfy_has_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request = request)

    client, http = _client(handler)
    with pytest.raises(ComfyUnavailableError, match = "Start ComfyUI"):
        _drive(client.check_reachable())
    _drive(http.aclose())


def test_poll_reports_comfy_generation_failure():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json = {
                "prompt-1": {
                    "status": {
                        "status_str": "error",
                        "completed": True,
                        "messages": [
                            ["execution_error", {"exception_message": "CUDA out of memory"}]
                        ],
                    }
                }
            },
        )

    client, http = _client(handler, poll_interval = 0)
    with pytest.raises(ComfyGenerationError, match = "CUDA out of memory"):
        _drive(client.wait_for_image("prompt-1"))
    _drive(http.aclose())


def test_poll_reports_completed_prompt_without_output():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json = {
                "prompt-1": {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {},
                }
            },
        )

    client, http = _client(handler, poll_interval = 0)
    with pytest.raises(ComfyGenerationError, match = "without a SaveImage output"):
        _drive(client.wait_for_image("prompt-1"))
    _drive(http.aclose())


def test_poll_timeout_is_bounded():
    client, http = _client(
        lambda _request: httpx.Response(200, json = {}),
        poll_interval = 0,
        generation_timeout = 0,
    )
    with pytest.raises(ComfyGenerationTimeout, match = "timed out"):
        _drive(client.wait_for_image("prompt-1"))
    _drive(http.aclose())


def test_generate_submits_finds_output_and_downloads_png():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            captured["prompt"] = json.loads(request.content)["prompt"]
            return httpx.Response(200, json = {"prompt_id": "prompt-1"})
        if request.url.path == "/history/prompt-1":
            return httpx.Response(
                200,
                json = {
                    "prompt-1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "7": {
                                "images": [
                                    {"filename": "out.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        assert request.url.path == "/view"
        assert request.url.params["filename"] == "out.png"
        return httpx.Response(200, content = PNG)

    client, http = _client(handler, poll_interval = 0)
    prompt_id, image = _drive(client.generate({"test": True}))
    assert prompt_id == "prompt-1"
    assert image == PNG
    assert captured["prompt"] == {"test": True}
    _drive(http.aclose())


def test_release_waits_for_idle_reissues_and_requires_released_reserved_vram():
    captured = []
    queue_remaining = iter([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    torch_reserved = iter([6_000_000_000, 0, 0, 0, 0])
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/system_stats":
            return httpx.Response(
                200,
                json = {"devices": [{"torch_vram_total": next(torch_reserved)}]},
            )
        if request.url.path == "/prompt":
            return httpx.Response(
                200,
                json = {"exec_info": {"queue_remaining": next(queue_remaining)}},
            )
        assert request.url.path == "/free"
        captured.append(json.loads(request.content))
        return httpx.Response(200)

    client, http = _client(handler, poll_interval = 0)
    _drive(client.release())
    assert captured == [
        {"unload_models": True, "free_memory": True},
        {"unload_models": True, "free_memory": True},
    ]
    assert paths[:3] == ["/prompt", "/prompt", "/free"]
    assert paths.count("/free") == 2
    _drive(http.aclose())


def test_release_accepts_an_already_unloaded_comfy():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json = {"devices": [{"torch_vram_total": 0}]})
        if request.url.path == "/prompt":
            return httpx.Response(200, json = {"exec_info": {"queue_remaining": 0}})
        return httpx.Response(200)

    client, http = _client(handler, poll_interval = 0)
    _drive(client.release())
    _drive(http.aclose())


def test_release_treats_cpu_only_comfy_as_having_no_reserved_vram():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(
                200,
                json = {
                    "devices": [
                        {"type": "cpu", "torch_vram_total": 64_000_000_000}
                    ]
                },
            )
        if request.url.path == "/prompt":
            return httpx.Response(200, json = {"exec_info": {"queue_remaining": 0}})
        return httpx.Response(200)

    client, http = _client(handler, poll_interval = 0)
    _drive(client.release())
    _drive(http.aclose())


def test_release_ignores_directml_placeholder_memory_statistics():
    free_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal free_requests
        if request.url.path == "/system_stats":
            return httpx.Response(
                200,
                json = {
                    "devices": [
                        {
                            "type": "privateuseone",
                            "name": "privateuseone:0",
                            "vram_total": 1024**3,
                            "vram_free": 1024**3,
                            "torch_vram_total": 1024**3,
                            "torch_vram_free": 1024**3,
                        }
                    ]
                },
            )
        if request.url.path == "/prompt":
            return httpx.Response(200, json = {"exec_info": {"queue_remaining": 0}})
        free_requests += 1
        return httpx.Response(200)

    client, http = _client(handler, poll_interval = 0)
    _drive(client.release())
    assert free_requests == 1
    _drive(http.aclose())


def test_release_rejects_constant_reserved_vram_with_a_bounded_timeout():
    free_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal free_requests
        if request.url.path == "/system_stats":
            return httpx.Response(
                200,
                json = {"devices": [{"torch_vram_total": 6_000_000_000}]},
            )
        if request.url.path == "/prompt":
            return httpx.Response(200, json = {"exec_info": {"queue_remaining": 0}})
        free_requests += 1
        return httpx.Response(200)

    client, http = _client(handler, poll_interval = 0, release_timeout = 0.01)
    with pytest.raises(ComfyReleaseTimeout, match = "did not release model memory"):
        _drive(client.release())
    assert free_requests == 1
    _drive(http.aclose())


def test_release_deadline_cancels_a_slow_initial_request():
    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json = {"exec_info": {"queue_remaining": 0}})

    client, http = _client(handler, poll_interval = 0, release_timeout = 0.01)
    with pytest.raises(ComfyReleaseTimeout, match = "did not release model memory"):
        _drive(client.release())
    _drive(http.aclose())
