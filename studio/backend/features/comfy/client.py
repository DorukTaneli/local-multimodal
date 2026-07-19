# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Minimal asynchronous client for the local ComfyUI HTTP API."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx


DEFAULT_COMFY_BASE_URL = "http://127.0.0.1:8188"
CONNECT_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 1.0
GENERATION_TIMEOUT_SECONDS = 30.0 * 60.0
RELEASE_TIMEOUT_SECONDS = 30.0
RELEASE_STABLE_POLLS = 3
RELEASE_TORCH_RESERVED_TOLERANCE_BYTES = 32 * 1024 * 1024
# CPU and MPS do not expose discrete VRAM here. torch-directml devices use
# privateuseone and ComfyUI reports a fixed 1 GiB placeholder for their memory
# statistics, so none of these values can prove that model memory is retained.
UNOBSERVABLE_VRAM_DEVICE_TYPES = {"cpu", "mps", "privateuseone"}


class ComfyError(RuntimeError):
    """Base error for ComfyUI communication and response failures."""


class ComfyUnavailableError(ComfyError):
    """ComfyUI could not be reached."""


class ComfyCheckpointError(ComfyError):
    """The workflow checkpoint is not installed in ComfyUI."""


class ComfyGenerationError(ComfyError):
    """ComfyUI rejected or failed a generation."""


class ComfyGenerationTimeout(ComfyGenerationError):
    """Generation did not finish within the configured deadline."""


class ComfyReleaseTimeout(ComfyError):
    """ComfyUI did not finish releasing model memory before the deadline."""


def comfy_base_url() -> str:
    return (os.environ.get("COMFY_BASE_URL") or DEFAULT_COMFY_BASE_URL).strip().rstrip("/")


class ComfyClient:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        generation_timeout: float = GENERATION_TIMEOUT_SECONDS,
        release_timeout: float = RELEASE_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url or comfy_base_url()).rstrip("/")
        self.poll_interval = poll_interval
        self.generation_timeout = generation_timeout
        self.release_timeout = release_timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url = self.base_url,
            timeout = httpx.Timeout(CONNECT_TIMEOUT_SECONDS),
            trust_env = False,
        )

    async def __aenter__(self) -> "ComfyClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.RequestError as exc:
            raise ComfyUnavailableError(
                f"Could not reach ComfyUI at {self.base_url}. Start ComfyUI and try again."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()[:500]
            suffix = f": {detail}" if detail else ""
            raise ComfyError(
                f"ComfyUI {path} returned HTTP {exc.response.status_code}{suffix}"
            ) from exc

    @staticmethod
    def _json(response: httpx.Response, description: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ComfyError(f"ComfyUI returned invalid JSON for {description}.") from exc

    async def check_reachable(self) -> None:
        response = await self._request("GET", "/system_stats")
        payload = self._json(response, "system status")
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI returned an invalid system status response.")

    def _release_timeout_error(self) -> ComfyReleaseTimeout:
        return ComfyReleaseTimeout(
            f"ComfyUI did not release model memory within "
            f"{self.release_timeout:g} seconds."
        )

    async def _request_before_deadline(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        **kwargs: Any,
    ) -> httpx.Response:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise self._release_timeout_error()
        kwargs["timeout"] = httpx.Timeout(
            remaining,
            connect = min(CONNECT_TIMEOUT_SECONDS, remaining),
        )
        try:
            return await asyncio.wait_for(
                self._request(method, path, **kwargs),
                timeout = remaining,
            )
        except ComfyUnavailableError as exc:
            if isinstance(exc.__cause__, httpx.TimeoutException):
                raise self._release_timeout_error() from exc
            raise
        except TimeoutError as exc:
            raise self._release_timeout_error() from exc

    async def _torch_vram_reserved_snapshot(self, *, deadline: float) -> tuple[int, ...]:
        response = await self._request_before_deadline(
            "GET", "/system_stats", deadline = deadline
        )
        payload = self._json(response, "system status")
        devices = payload.get("devices") if isinstance(payload, dict) else None
        if not isinstance(devices, list) or not devices:
            raise ComfyError("ComfyUI returned invalid device memory statistics.")
        reserved_values: list[int] = []
        for device in devices:
            if not isinstance(device, dict):
                raise ComfyError("ComfyUI returned invalid device memory statistics.")
            if device.get("type") in UNOBSERVABLE_VRAM_DEVICE_TYPES:
                continue
            value = device.get("torch_vram_total")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ComfyError("ComfyUI returned invalid device memory statistics.")
            reserved_values.append(int(value))
        return tuple(reserved_values)

    async def _queue_remaining(self, *, deadline: float) -> int:
        response = await self._request_before_deadline(
            "GET", "/prompt", deadline = deadline
        )
        payload = self._json(response, "queue status")
        exec_info = payload.get("exec_info") if isinstance(payload, dict) else None
        remaining = exec_info.get("queue_remaining") if isinstance(exec_info, dict) else None
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
            raise ComfyError("ComfyUI returned an invalid queue status response.")
        return remaining

    @staticmethod
    def _model_memory_is_released(reserved: tuple[int, ...]) -> bool:
        return all(
            value <= RELEASE_TORCH_RESERVED_TOLERANCE_BYTES for value in reserved
        )

    async def _sleep_before_deadline(self, deadline: float) -> None:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise self._release_timeout_error()
        await asyncio.sleep(min(self.poll_interval, remaining))

    async def _wait_for_idle_queue(self, *, deadline: float) -> None:
        while await self._queue_remaining(deadline = deadline) != 0:
            await self._sleep_before_deadline(deadline)

    async def available_checkpoints(self) -> list[str]:
        response = await self._request("GET", "/models/checkpoints")
        payload = self._json(response, "checkpoint list")
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ComfyError("ComfyUI returned an invalid checkpoint list.")
        return payload

    async def preflight(self, checkpoint: str) -> None:
        await self.check_reachable()
        checkpoints = await self.available_checkpoints()
        if checkpoint not in checkpoints:
            raise ComfyCheckpointError(
                f"ComfyUI checkpoint {checkpoint!r} is not installed. "
                "Add it to ComfyUI's checkpoints folder and refresh ComfyUI."
            )

    async def submit(self, workflow: dict[str, Any]) -> str:
        try:
            response = await self._request("POST", "/prompt", json = {"prompt": workflow})
        except ComfyUnavailableError:
            raise
        except ComfyError as exc:
            raise ComfyGenerationError(str(exc)) from exc
        payload = self._json(response, "prompt submission")
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyGenerationError("ComfyUI did not return a prompt ID.")
        return prompt_id

    @staticmethod
    def _failure_detail(record: dict[str, Any]) -> str:
        status = record.get("status")
        messages = status.get("messages", []) if isinstance(status, dict) else []
        for message in reversed(messages) if isinstance(messages, list) else []:
            if isinstance(message, (list, tuple)) and len(message) > 1 and isinstance(message[1], dict):
                detail = message[1].get("exception_message") or message[1].get("message")
                if detail:
                    return str(detail)
        return "ComfyUI reported that image generation failed."

    @staticmethod
    def _saved_image(record: dict[str, Any], save_node_id: str) -> dict[str, str] | None:
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            return None
        output = outputs.get(save_node_id)
        images = output.get("images") if isinstance(output, dict) else None
        if not isinstance(images, list):
            return None
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if isinstance(filename, str) and filename:
                return {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
        return None

    async def wait_for_image(self, prompt_id: str, *, save_node_id: str = "7") -> dict[str, str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.generation_timeout
        while True:
            if loop.time() >= deadline:
                raise ComfyGenerationTimeout(
                    f"ComfyUI generation timed out after {self.generation_timeout:g} seconds."
                )
            response = await self._request("GET", f"/history/{prompt_id}")
            payload = self._json(response, "generation history")
            if not isinstance(payload, dict):
                raise ComfyGenerationError("ComfyUI returned invalid generation history.")
            record = payload.get(prompt_id)
            if isinstance(record, dict):
                status = record.get("status")
                status_str = status.get("status_str") if isinstance(status, dict) else None
                completed = bool(status.get("completed")) if isinstance(status, dict) else False
                if status_str == "error":
                    raise ComfyGenerationError(self._failure_detail(record))
                image = self._saved_image(record, save_node_id)
                if image is not None:
                    return image
                if completed or status_str == "success":
                    raise ComfyGenerationError(
                        f"ComfyUI completed prompt {prompt_id} without a SaveImage output."
                    )
            await asyncio.sleep(self.poll_interval)

    async def download_image(self, image: dict[str, str]) -> bytes:
        response = await self._request("GET", "/view", params = image)
        content = response.content
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ComfyGenerationError("ComfyUI returned an invalid PNG image.")
        return content

    async def generate(
        self, workflow: dict[str, Any], *, save_node_id: str = "7"
    ) -> tuple[str, bytes]:
        prompt_id = await self.submit(workflow)
        image = await self.wait_for_image(prompt_id, save_node_id = save_node_id)
        return prompt_id, await self.download_image(image)

    async def release(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.release_timeout
        while True:
            # ComfyUI consumes /free flags between queued jobs. Posting while a
            # backlog exists could unload only for the next job to reload the
            # checkpoint, so establish an idle boundary first.
            await self._wait_for_idle_queue(deadline = deadline)
            await self._request_before_deadline(
                "POST",
                "/free",
                deadline = deadline,
                json = {"unload_models": True, "free_memory": True},
            )

            released_polls = 0
            while True:
                await self._sleep_before_deadline(deadline)
                if await self._queue_remaining(deadline = deadline) != 0:
                    # New work won the race after the idle check. Let it drain,
                    # then issue a fresh /free after its models are no longer in use.
                    break
                reserved = await self._torch_vram_reserved_snapshot(deadline = deadline)
                if self._model_memory_is_released(reserved):
                    released_polls += 1
                    if released_polls >= RELEASE_STABLE_POLLS:
                        # Recheck both signals before handing VRAM to llama.cpp.
                        if await self._queue_remaining(deadline = deadline) != 0:
                            break
                        final_reserved = await self._torch_vram_reserved_snapshot(
                            deadline = deadline
                        )
                        if (
                            self._model_memory_is_released(final_reserved)
                            and await self._queue_remaining(deadline = deadline) == 0
                        ):
                            return
                        break
                else:
                    released_polls = 0
