// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { COMFY_TOOL_NAME, type ComfyAsset } from "./types.ts";

function isAsset(value: unknown): value is ComfyAsset {
  if (!value || typeof value !== "object") return false;
  const asset = value as Partial<ComfyAsset>;
  return (
    typeof asset.assetId === "string" &&
    asset.assetId.length > 0 &&
    typeof asset.seed === "number" &&
    Number.isFinite(asset.seed) &&
    typeof asset.width === "number" &&
    asset.width > 0 &&
    typeof asset.height === "number" &&
    asset.height > 0 &&
    asset.mimeType === "image/png"
  );
}

export function extractComfyContextProjection(part: unknown): string | null {
  if (!part || typeof part !== "object") return null;
  const candidate = part as {
    type?: unknown;
    toolName?: unknown;
    result?: unknown;
  };
  if (candidate.type !== "tool-call" || candidate.toolName !== COMFY_TOOL_NAME) {
    return null;
  }
  if (!candidate.result || typeof candidate.result !== "object") return null;
  const result = candidate.result as {
    effectivePrompt?: unknown;
    assets?: unknown;
    contextProjection?: unknown;
  };
  if (
    typeof result.effectivePrompt !== "string" ||
    result.effectivePrompt.trim().length === 0 ||
    !Array.isArray(result.assets) ||
    result.assets.length === 0 ||
    !result.assets.every(isAsset) ||
    typeof result.contextProjection !== "string"
  ) {
    return null;
  }
  const projection = result.contextProjection.trim();
  return /^\[Image: .+\]$/.test(projection) ? projection : null;
}

export function buildContextProjection(
  conversationTags: string,
  imageTags: string,
): string {
  const tags = [conversationTags, imageTags]
    .map((value) => value.trim().replace(/\s+/g, " "))
    .filter(Boolean)
    .join(", ");
  return `[Image: ${tags}]`;
}
