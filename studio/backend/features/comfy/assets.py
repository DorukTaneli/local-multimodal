# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UUID-addressed PNG storage for generated ComfyUI assets."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from uuid import UUID, uuid4

from utils.paths.storage_roots import assets_root


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AssetError(RuntimeError):
    """A generated asset is invalid, missing, or unsafe to serve."""


def comfy_assets_root() -> Path:
    return assets_root() / "comfy"


def save_png(content: bytes) -> UUID:
    if not content.startswith(PNG_SIGNATURE):
        raise AssetError("Generated asset is not a valid PNG image.")
    root = comfy_assets_root()
    asset_id = uuid4()
    path = root / f"{asset_id}.png"
    try:
        root.mkdir(parents = True, exist_ok = True)
        with path.open("xb") as output:
            output.write(content)
    except OSError as exc:
        raise AssetError(f"Could not save generated image: {exc}") from exc
    return asset_id


def resolve_png_asset(asset_id: str) -> Path:
    try:
        parsed = UUID(asset_id)
    except (ValueError, AttributeError) as exc:
        raise AssetError("Asset ID must be a valid UUID.") from exc

    try:
        root = comfy_assets_root().resolve()
        candidate = root / f"{parsed}.png"
        candidate_mode = os.lstat(candidate).st_mode
        if not stat.S_ISREG(candidate_mode):
            raise AssetError("Generated image was not found.")
        resolved = candidate.resolve(strict = True)
        resolved.relative_to(root)
        with resolved.open("rb") as image:
            signature = image.read(len(PNG_SIGNATURE))
    except (OSError, ValueError) as exc:
        raise AssetError("Generated image was not found.") from exc
    if resolved.suffix.lower() != ".png" or signature != PNG_SIGNATURE:
        raise AssetError("Generated image was not found.")
    return resolved
