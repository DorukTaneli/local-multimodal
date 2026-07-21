// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const COMFY_TOOL_NAME = "comfy_generate_image";

export type ComfyAsset = {
  assetId: string;
  seed: number;
  width: number;
  height: number;
  mimeType: "image/png";
};

export type ComfyStatus = {
  ready: boolean;
  reachable: boolean;
  workflowValid: boolean;
  checkpointAvailable: boolean;
  busy: boolean;
  checkpoint: string | null;
  failures: string[];
};

export type ComfyGenerationResult = {
  effectivePrompt: string;
  asset: ComfyAsset;
};

export type ComfyToolResult = {
  effectivePrompt: string;
  assets: ComfyAsset[];
  contextProjection: string;
};

export type ComfyToolPart = {
  type: "tool-call";
  toolName: typeof COMFY_TOOL_NAME;
  toolCallId: string;
  args: { imageTags: string };
  argsText: string;
  result: ComfyToolResult;
};
