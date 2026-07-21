// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { saveChatMessage } from "../chat/api/chat-api";
import type { ExportedMessageRepository } from "@assistant-ui/react";
import {
  COMFY_TOOL_NAME,
  type ComfyAsset,
  type ComfyToolPart,
  type ComfyToolResult,
} from "./types";

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (repository: ExportedMessageRepository) => void;
};

async function updatePersistedMessage(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string;
  update: (content: readonly unknown[]) => unknown[];
}): Promise<void> {
  const original = args.thread.export();
  const entry = original.messages.find(({ message }) => message.id === args.messageId);
  if (!entry || entry.message.role !== "assistant" || !Array.isArray(entry.message.content)) {
    throw new Error("The assistant response is no longer available.");
  }
  const content = args.update(entry.message.content) as typeof entry.message.content;
  const messages = original.messages.map((item) =>
    item.message.id === args.messageId
      ? { ...item, message: { ...item.message, content } }
      : item,
  ) as typeof original.messages;
  args.thread.import({ ...original, messages });
  try {
    await saveChatMessage({
      id: args.messageId,
      threadId: args.remoteId,
      parentId: entry.parentId,
      role: "assistant",
      content,
      createdAt: entry.message.createdAt?.getTime?.() ?? Date.now(),
    });
  } catch (error) {
    args.thread.import(original);
    throw error;
  }
}

export async function appendComfyToolPart(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string;
  imageTags: string;
  result: ComfyToolResult;
}): Promise<void> {
  const toolCallId = `comfy_${crypto.randomUUID()}`;
  const toolArgs = { imageTags: args.imageTags };
  const part: ComfyToolPart = {
    type: "tool-call",
    toolName: COMFY_TOOL_NAME,
    toolCallId,
    args: toolArgs,
    argsText: JSON.stringify(toolArgs),
    result: args.result,
  };
  await updatePersistedMessage({
    ...args,
    update: (content) => [...content, part],
  });
}

export async function replaceComfyAsset(args: {
  thread: ThreadImportExport;
  messageId: string;
  remoteId: string;
  toolCallId: string;
  asset: ComfyAsset;
}): Promise<void> {
  await updatePersistedMessage({
    ...args,
    update: (content) => {
      let updated = false;
      const next = content.map((part) => {
        if (!part || typeof part !== "object") return part;
        const candidate = part as Partial<ComfyToolPart>;
        if (
          candidate.type !== "tool-call" ||
          candidate.toolName !== COMFY_TOOL_NAME ||
          candidate.toolCallId !== args.toolCallId ||
          !candidate.result ||
          !Array.isArray(candidate.result.assets)
        ) {
          return part;
        }
        updated = true;
        return {
          ...candidate,
          result: {
            ...candidate.result,
            assets: [args.asset],
          },
        };
      });
      if (!updated) throw new Error("The generated image record is no longer available.");
      return next;
    },
  });
}

const conversationTagsKey = (threadId: string) =>
  `local-multimodal:conversation-tags:${threadId}`;

export function loadConversationTags(threadId: string): string {
  try {
    return localStorage.getItem(conversationTagsKey(threadId)) ?? "";
  } catch {
    return "";
  }
}

export function saveConversationTags(threadId: string, tags: string): void {
  try {
    localStorage.setItem(conversationTagsKey(threadId), tags);
  } catch {
    // Generation remains usable when browser storage is unavailable.
  }
}
