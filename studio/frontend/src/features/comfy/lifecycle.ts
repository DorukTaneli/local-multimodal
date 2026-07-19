// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { releaseComfy } from "./api";

export async function releaseComfyBeforeLocalModelLoad(): Promise<void> {
  try {
    // The backend returns only after ComfyUI's queue is idle and its asynchronous
    // /free request has released the framework's reserved VRAM.
    await releaseComfy();
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Cannot load the chat model: ${error.message}`);
    }
    throw new Error("Cannot load the chat model while ComfyUI is busy.");
  }
}
