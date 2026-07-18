// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useContext } from "react";
import { ComfyModelLifecycleContext } from "./lifecycle-context";

export function useComfyModelLifecycle(): () => Promise<boolean> {
  const ejectModel = useContext(ComfyModelLifecycleContext);
  if (!ejectModel) {
    throw new Error("Comfy image generation is outside the chat model runtime.");
  }
  return ejectModel;
}
