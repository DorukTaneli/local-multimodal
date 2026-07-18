# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    imageTags: str | None = Field(default = None, max_length = 100_000)
    conversationTags: str | None = Field(default = None, max_length = 100_000)
    effectivePrompt: str | None = Field(default = None, max_length = 200_000)


class GeneratedAsset(BaseModel):
    assetId: str
    seed: int
    width: int
    height: int
    mimeType: str = "image/png"


class GenerateResponse(BaseModel):
    effectivePrompt: str
    asset: GeneratedAsset


class ReleaseResponse(BaseModel):
    released: bool = True
    offline: bool = False


class StatusResponse(BaseModel):
    ready: bool
    reachable: bool
    workflowValid: bool
    checkpointAvailable: bool
    busy: bool
    checkpoint: str | None = None
    failures: list[str] = Field(default_factory = list)
