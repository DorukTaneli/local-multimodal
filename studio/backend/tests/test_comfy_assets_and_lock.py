# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import HTTPException
import pytest

from features.comfy import assets as assets_module
from features.comfy.assets import AssetError, PNG_SIGNATURE, resolve_png_asset, save_png
from features.comfy.models import GenerateRequest
import features.comfy.router as router_module
from features.comfy import workflow as workflow_module


def test_uuid_asset_storage_and_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(assets_module, "assets_root", lambda: tmp_path / "assets")
    asset_id = save_png(PNG_SIGNATURE + b"payload")

    path = resolve_png_asset(str(asset_id))
    assert path == (tmp_path / "assets" / "comfy" / f"{asset_id}.png").resolve()
    assert path.read_bytes() == PNG_SIGNATURE + b"payload"


@pytest.mark.parametrize("asset_id", ["../secret", "not-a-uuid", f"{uuid4()}.png"])
def test_asset_resolution_rejects_traversal_and_non_uuid_ids(tmp_path, monkeypatch, asset_id):
    monkeypatch.setattr(assets_module, "assets_root", lambda: tmp_path / "assets")
    with pytest.raises(AssetError, match = "valid UUID"):
        resolve_png_asset(asset_id)


def test_missing_uuid_asset_is_not_served(tmp_path, monkeypatch):
    monkeypatch.setattr(assets_module, "assets_root", lambda: tmp_path / "assets")
    with pytest.raises(AssetError, match = "not found"):
        resolve_png_asset(str(uuid4()))


def test_non_png_file_with_uuid_name_is_not_served(tmp_path, monkeypatch):
    monkeypatch.setattr(assets_module, "assets_root", lambda: tmp_path / "assets")
    asset_id = uuid4()
    path = tmp_path / "assets" / "comfy" / f"{asset_id}.png"
    path.parent.mkdir(parents = True)
    path.write_bytes(b"not a png")
    with pytest.raises(AssetError, match = "not found"):
        resolve_png_asset(str(asset_id))


def test_release_is_rejected_promptly_while_generation_slot_is_owned():
    async def scenario():
        async with router_module.operation_slot():
            with pytest.raises(HTTPException) as exc_info:
                await router_module.release()
            assert exc_info.value.status_code == 409
            assert "busy" in str(exc_info.value.detail).lower()

    asyncio.run(scenario())


def test_release_treats_offline_comfy_as_already_released(monkeypatch):
    class OfflineClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def release(self):
            raise router_module.ComfyUnavailableError("offline")

    monkeypatch.setattr(router_module, "ComfyClient", OfflineClient)
    response = asyncio.run(router_module.release())
    assert response.released is True
    assert response.offline is True


def test_generate_preflights_without_releasing_models(monkeypatch):
    workflow_path = (
        workflow_module.REPOSITORY_ROOT / "comfy" / workflow_module.WORKFLOW_FILENAME
    )
    workflow = json.loads(workflow_path.read_text(encoding = "utf-8"))
    calls: list[str] = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def preflight(self, checkpoint):
            calls.append(f"preflight:{checkpoint}")

        async def generate(self, submitted, *, save_node_id):
            calls.append(f"generate:{save_node_id}")
            assert submitted["2"]["inputs"]["text"].endswith("night, 1girl, wink")
            return "prompt-1", PNG_SIGNATURE + b"payload"

        async def release(self):
            calls.append("release")

    asset_id = uuid4()
    monkeypatch.setattr(router_module, "ComfyClient", FakeClient)
    monkeypatch.setattr(router_module, "load_workflow", lambda: workflow)
    monkeypatch.setattr(router_module, "save_png", lambda _png: asset_id)

    response = asyncio.run(
        router_module.generate(
            GenerateRequest(imageTags = "1girl, wink", conversationTags = "night")
        )
    )
    assert calls == [
        "preflight:prefectIllustriousXL_v70.safetensors",
        "generate:7",
    ]
    assert response.asset.assetId == str(asset_id)
    assert (response.asset.width, response.asset.height) == (896, 1152)
    assert response.asset.mimeType == "image/png"


def test_status_reports_preflight_and_busy_state(monkeypatch):
    workflow_path = (
        workflow_module.REPOSITORY_ROOT / "comfy" / workflow_module.WORKFLOW_FILENAME
    )
    workflow = json.loads(workflow_path.read_text(encoding = "utf-8"))

    class ReadyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def check_reachable(self):
            return None

        async def available_checkpoints(self):
            return ["prefectIllustriousXL_v70.safetensors"]

    monkeypatch.setattr(router_module, "ComfyClient", ReadyClient)
    monkeypatch.setattr(router_module, "load_workflow", lambda: workflow)

    ready = asyncio.run(router_module.status())
    assert ready.ready is True
    assert ready.reachable is True
    assert ready.workflowValid is True
    assert ready.checkpointAvailable is True

    async def busy_scenario():
        async with router_module.operation_slot():
            return await router_module.status()

    busy = asyncio.run(busy_scenario())
    assert busy.ready is False
    assert busy.busy is True
    assert any("busy" in failure.lower() for failure in busy.failures)


def test_router_exposes_exact_comfy_interfaces():
    assert {(route.path, next(iter(route.methods))) for route in router_module.router.routes} == {
        ("/status", "GET"),
        ("/generate", "POST"),
        ("/release", "POST"),
        ("/assets/{asset_id}", "GET"),
    }
