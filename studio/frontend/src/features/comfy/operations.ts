// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComfyAsset, ComfyGenerationResult, ComfyStatus } from "./types.ts";

export async function runGenerationAfterReview(deps: {
  status: () => Promise<ComfyStatus>;
  requireReady: (status: ComfyStatus) => void;
  eject: () => Promise<boolean>;
  generate: () => Promise<ComfyGenerationResult>;
  persist: (result: ComfyGenerationResult) => Promise<void>;
}): Promise<void> {
  const status = await deps.status();
  deps.requireReady(status);
  if (!(await deps.eject())) {
    throw new Error("The local chat model could not be unloaded.");
  }
  const result = await deps.generate();
  await deps.persist(result);
}

export async function runComfyReroll(deps: {
  status: () => Promise<ComfyStatus>;
  requireReady: (status: ComfyStatus) => void;
  shouldEject: () => boolean;
  eject: () => Promise<boolean>;
  generate: () => Promise<ComfyGenerationResult>;
  persistAsset: (asset: ComfyAsset) => Promise<void>;
}): Promise<void> {
  const status = await deps.status();
  deps.requireReady(status);
  if (deps.shouldEject() && !(await deps.eject())) {
    throw new Error("The local chat model could not be unloaded.");
  }
  const result = await deps.generate();
  await deps.persistAsset(result.asset);
}
