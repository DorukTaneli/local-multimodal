// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComfyAsset, ComfyGenerationResult, ComfyStatus } from "./types.ts";

export async function runGenerationAfterReview(deps: {
  status: () => Promise<ComfyStatus>;
  requireReady: (status: ComfyStatus) => void;
  generate: () => Promise<ComfyGenerationResult>;
  persist: (result: ComfyGenerationResult) => Promise<void>;
  discard: (asset: ComfyAsset) => Promise<void>;
}): Promise<void> {
  const status = await deps.status();
  deps.requireReady(status);
  const result = await deps.generate();
  try {
    await deps.persist(result);
  } catch (error) {
    await deps.discard(result.asset).catch(() => undefined);
    throw error;
  }
}

export async function runComfyReroll(deps: {
  status: () => Promise<ComfyStatus>;
  requireReady: (status: ComfyStatus) => void;
  generate: () => Promise<ComfyGenerationResult>;
  persistAsset: (asset: ComfyAsset) => Promise<void>;
  discard: (asset: ComfyAsset) => Promise<void>;
}): Promise<ComfyAsset> {
  const status = await deps.status();
  deps.requireReady(status);
  const result = await deps.generate();
  try {
    await deps.persistAsset(result.asset);
  } catch (error) {
    await deps.discard(result.asset).catch(() => undefined);
    throw error;
  }
  return result.asset;
}
