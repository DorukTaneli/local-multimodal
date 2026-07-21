// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  ComfyGenerationResult,
  ComfyStatus,
} from "./types";

export class ComfyApiError extends Error {
  readonly status: number;

  constructor(
    message: string,
    status: number,
  ) {
    super(message);
    this.name = "ComfyApiError";
    this.status = status;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body && typeof body === "object"
        ? formatFastApiDetail((body as { detail?: unknown }).detail)
        : "";
    throw new ComfyApiError(detail || `ComfyUI request failed (${response.status})`, response.status);
  }
  return body as T;
}

export async function getComfyStatus(): Promise<ComfyStatus> {
  return parseResponse<ComfyStatus>(await authFetch("/api/comfy/status"));
}

export async function generateComfyImage(
  body:
    | { imageTags: string; conversationTags: string }
    | { effectivePrompt: string },
): Promise<ComfyGenerationResult> {
  return parseResponse<ComfyGenerationResult>(
    await authFetch("/api/comfy/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function deleteComfyAsset(assetId: string): Promise<void> {
  const response = await authFetch(
    `/api/comfy/assets/${encodeURIComponent(assetId)}`,
    { method: "DELETE" },
  );
  if (!response.ok && response.status !== 404) {
    throw new ComfyApiError(
      `Could not delete generated image (${response.status})`,
      response.status,
    );
  }
}

export function requireReadyComfy(status: ComfyStatus): void {
  if (status.ready && !status.busy) return;
  const message = status.busy
    ? "ComfyUI is busy generating an image. Wait for it to finish."
    : status.failures.filter(Boolean).join(" ") || "ComfyUI is not ready.";
  throw new Error(message);
}
