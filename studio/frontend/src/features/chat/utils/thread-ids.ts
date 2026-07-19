// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ASSISTANT_LOCAL_THREAD_ID_PREFIX = "__LOCALID_";
const incognitoThreadIds = new Set<string>();

export function isAssistantLocalThreadId(
  threadId: string | null | undefined,
): boolean {
  return Boolean(threadId?.startsWith(ASSISTANT_LOCAL_THREAD_ID_PREFIX));
}

// `__LOCALID_` identifies assistant-ui's client-generated ids, not storage
// state: saved threads intentionally retain that id after initialization.
// Incognito identity is tracked separately at creation and stays attached to
// the thread if the live toggle changes. Persistence reads and writes consult
// this marker, so an in-flight temporary chat cannot leak into history.
export function markThreadIncognito(threadId: string): void {
  incognitoThreadIds.add(threadId);
}

export function isThreadIncognito(threadId: string): boolean {
  return incognitoThreadIds.has(threadId);
}
