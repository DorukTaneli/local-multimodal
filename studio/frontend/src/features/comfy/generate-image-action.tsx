// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { useChatRuntimeStore } from "../chat/stores/chat-runtime-store";
import { toast } from "@/lib/toast";
import { useAui, useAuiState } from "@assistant-ui/react";
import { ImagePlusIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  deleteComfyAsset,
  generateComfyImage,
  getComfyStatus,
  requireReadyComfy,
} from "./api";
import { generateHiddenImageTags } from "./hidden-tags";
import { runGenerationAfterReview } from "./operations";
import {
  appendComfyToolPart,
  loadConversationTags,
  saveConversationTags,
} from "./persistence";
import { buildContextProjection } from "./serialization";
import { isComfyGenerationEligible } from "./eligibility";

function visibleAssistantText(content: readonly unknown[]): string {
  return content
    .flatMap((part) =>
      part &&
      typeof part === "object" &&
      (part as { type?: unknown }).type === "text" &&
      typeof (part as { text?: unknown }).text === "string"
        ? [(part as { text: string }).text]
        : [],
    )
    .join("\n")
    .trim();
}

export function ComfyGenerateImageAction() {
  const aui = useAui();
  const messageId = useAuiState(({ message }) => message.id);
  const content = useAuiState(({ message }) => message.content);
  const threadRunning = useAuiState(({ thread }) => thread.isRunning);
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const checkpoint = useChatRuntimeStore((state) => state.params.checkpoint);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const incognito = useChatRuntimeStore((state) => state.incognito);
  const modelLoading = useChatRuntimeStore((state) => state.modelLoading);
  const [phase, setPhase] = useState<"tagging" | "generating" | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [imageTags, setImageTags] = useState("");
  const [conversationTags, setConversationTags] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const text = visibleAssistantText(Array.isArray(content) ? content : []);
  const eligible = isComfyGenerationEligible({
    checkpoint,
    incognito,
    modelLoading,
    threadRunning,
    remoteId,
    activeThreadId,
    visibleText: text,
  });

  useEffect(() => () => abortRef.current?.abort(), []);

  const start = async () => {
    if (!eligible || !remoteId) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase("tagging");
    try {
      const tags = await generateHiddenImageTags(text, controller.signal);
      setImageTags(tags);
      setConversationTags(loadConversationTags(remoteId));
      setDialogOpen(true);
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error("Could not prepare image tags", {
          description: error instanceof Error ? error.message : undefined,
        });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setPhase(null);
    }
  };

  const generate = async () => {
    if (!remoteId || !imageTags.trim()) return;
    setPhase("generating");
    saveConversationTags(remoteId, conversationTags);
    try {
      await runGenerationAfterReview({
        status: getComfyStatus,
        requireReady: requireReadyComfy,
        generate: () =>
          generateComfyImage({
            imageTags: imageTags.trim(),
            conversationTags: conversationTags.trim(),
          }),
        persist: async (generated) => {
          await appendComfyToolPart({
            thread: {
              export: () => aui.thread().export(),
              import: (repository) => aui.thread().import(repository),
            },
            messageId,
            remoteId,
            imageTags: imageTags.trim(),
            result: {
              effectivePrompt: generated.effectivePrompt,
              assets: [generated.asset],
              contextProjection: buildContextProjection(
                conversationTags,
                imageTags,
              ),
            },
          });
        },
        discard: (asset) => deleteComfyAsset(asset.assetId),
      });
      setDialogOpen(false);
    } catch (error) {
      toast.error("Image generation failed", {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setPhase(null);
    }
  };

  if (!eligible && phase === null && !dialogOpen) return null;

  return (
    <>
      <TooltipIconButton
        tooltip={phase === "tagging" ? "Preparing image tags…" : "Generate image"}
        disabled={!eligible || phase !== null}
        onClick={() => void start()}
      >
        <ImagePlusIcon className="size-icon" strokeWidth={1.75} />
      </TooltipIconButton>
      <Dialog
        open={dialogOpen}
        onOpenChange={(open) => phase !== "generating" && setDialogOpen(open)}
      >
        <DialogContent showCloseButton={false}>
          <DialogTitle className="sr-only">Generate image</DialogTitle>
          <label className="grid gap-2 font-medium">
            Image tags
            <Textarea
              value={imageTags}
              onChange={(event) => setImageTags(event.target.value)}
              disabled={phase === "generating"}
              rows={5}
            />
          </label>
          <label className="grid gap-2 font-medium">
            Conversation tags
            <Textarea
              value={conversationTags}
              onChange={(event) => setConversationTags(event.target.value)}
              disabled={phase === "generating"}
              rows={3}
            />
          </label>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={phase === "generating"}
              onClick={() => setDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={phase === "generating" || !imageTags.trim()}
              onClick={() => void generate()}
            >
              {phase === "generating" ? "Generating…" : "Generate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
