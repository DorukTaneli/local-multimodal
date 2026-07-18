// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { streamChatCompletions } from "../chat/api/chat-api";
import { useChatRuntimeStore } from "../chat/stores/chat-runtime-store";
import type { OpenAIChatCompletionsRequest } from "../chat/types/api";

const TAG_INSTRUCTION =
  "Convert the supplied scene into comma-separated Illustrious-style visual tags only. Return tags only: no prose, markdown, lists, or explanation.";

export async function generateHiddenImageTags(
  assistantText: string,
  signal: AbortSignal,
): Promise<string> {
  const store = useChatRuntimeStore.getState();
  const { params } = store;
  if (!params.checkpoint) throw new Error("Load a local chat model first.");

  const payload: OpenAIChatCompletionsRequest = {
    model: params.checkpoint,
    messages: [
      { role: "system", content: TAG_INSTRUCTION },
      { role: "user", content: assistantText },
    ],
    stream: true,
    temperature: params.temperature,
    top_p: params.topP,
    max_tokens: params.maxTokens,
    top_k: params.topK,
    min_p: params.minP,
    repetition_penalty: params.repetitionPenalty,
    presence_penalty: params.presencePenalty,
    enable_tools: false,
    ...(store.supportsReasoning
      ? store.reasoningStyle === "enable_thinking_effort"
        ? { enable_thinking: false }
        : store.reasoningStyle === "reasoning_effort"
          ? store.supportsReasoningOff
            ? { reasoning_effort: "none" as const }
            : {}
          : { thinking: { type: "disabled" as const } }
      : {}),
  };

  let tags = "";
  for await (const chunk of streamChatCompletions(payload, signal)) {
    const content = chunk.choices?.[0]?.delta?.content;
    if (typeof content === "string") tags += content;
  }
  const cleaned = tags.trim().replace(/^```[^\n]*\n?|```$/g, "").trim();
  if (!cleaned) throw new Error("The chat model did not return image tags.");
  return cleaned;
}
