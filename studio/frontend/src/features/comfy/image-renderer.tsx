// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useChatRuntimeStore } from "../chat/stores/chat-runtime-store";
import { useAui, useAuiState, type ToolCallMessagePartComponent } from "@assistant-ui/react";
import { RefreshCwIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import {
  deleteComfyAsset,
  generateComfyImage,
  getComfyStatus,
  requireReadyComfy,
} from "./api";
import { runComfyReroll } from "./operations";
import { replaceComfyAsset } from "./persistence";
import { COMFY_TOOL_NAME, type ComfyAsset, type ComfyToolResult } from "./types";

function ComfyAssetImage({ asset }: { asset: ComfyAsset }) {
  const [source, setSource] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let disposed = false;
    let objectUrl: string | null = null;
    void authFetch(`/api/comfy/assets/${encodeURIComponent(asset.assetId)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Image request failed (${response.status})`);
        objectUrl = URL.createObjectURL(await response.blob());
        if (!disposed) setSource(objectUrl);
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset.assetId]);

  if (failed) {
    return (
      <div className="flex min-h-40 items-center justify-center rounded-2xl bg-muted text-sm text-muted-foreground">
        Image unavailable
      </div>
    );
  }
  if (!source) {
    return (
      <div className="flex min-h-40 items-center justify-center rounded-2xl bg-muted/50">
        <Spinner />
      </div>
    );
  }
  return (
    <img
      src={source}
      alt="Locally generated image"
      width={asset.width}
      height={asset.height}
      className="h-auto max-h-[620px] w-auto max-w-full rounded-2xl object-contain"
    />
  );
}

const ComfyImageRendererImpl: ToolCallMessagePartComponent = (props) => {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const messageContent = useAuiState(({ message }) => message.content);
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const threadRunning = useAuiState(({ thread }) => thread.isRunning);
  const modelLoading = useChatRuntimeStore((state) => state.modelLoading);
  const [rerolling, setRerolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const result = props.result as ComfyToolResult | undefined;
  const toolCallIdFromProps = (props as { toolCallId?: unknown }).toolCallId;
  const toolCallId = (() => {
    if (typeof toolCallIdFromProps === "string") return toolCallIdFromProps;
    if (!Array.isArray(messageContent)) return null;
    const match = messageContent.find((part) => {
      if (!part || typeof part !== "object") return false;
      const candidate = part as { toolName?: unknown; result?: unknown };
      return candidate.toolName === COMFY_TOOL_NAME && candidate.result === props.result;
    }) as { toolCallId?: unknown } | undefined;
    return typeof match?.toolCallId === "string" ? match.toolCallId : null;
  })();

  if (
    !result ||
    typeof result.effectivePrompt !== "string" ||
    !Array.isArray(result.assets) ||
    result.assets.length === 0
  ) {
    return null;
  }

  const reroll = async () => {
    if (threadRunning || modelLoading) {
      setError("Wait for the chat response or model load to finish before rerolling.");
      return;
    }
    if (!remoteId || !toolCallId) {
      setError("This image is not attached to a saved chat.");
      return;
    }
    setRerolling(true);
    setError(null);
    try {
      const replacement = await runComfyReroll({
        status: getComfyStatus,
        requireReady: requireReadyComfy,
        generate: () =>
          generateComfyImage({ effectivePrompt: result.effectivePrompt }),
        persistAsset: (asset) =>
          replaceComfyAsset({
            thread: {
              export: () => aui.thread().export(),
              import: (repository) => aui.thread().import(repository),
            },
            messageId,
            remoteId,
            toolCallId,
            asset,
          }),
        discard: (asset) => deleteComfyAsset(asset.assetId),
      });
      await Promise.allSettled(
        result.assets
          .filter((asset) => asset.assetId !== replacement.assetId)
          .map((asset) => deleteComfyAsset(asset.assetId)),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Reroll failed.");
    } finally {
      setRerolling(false);
    }
  };

  return (
    <div className="my-3 grid gap-3">
      <div className="flex flex-wrap items-start gap-3">
        {result.assets.map((asset) => (
          <ComfyAssetImage key={asset.assetId} asset={asset} />
        ))}
      </div>
      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : null}
      <div>
        <Button
          size="sm"
          variant="outline"
          disabled={rerolling || threadRunning || modelLoading}
          onClick={() => void reroll()}
        >
          {rerolling ? <Spinner /> : <RefreshCwIcon className="size-4" />}
          {rerolling ? "Rerolling…" : "Reroll"}
        </Button>
      </div>
    </div>
  );
};

export const ComfyImageRenderer = memo(
  ComfyImageRendererImpl,
) as unknown as ToolCallMessagePartComponent;
