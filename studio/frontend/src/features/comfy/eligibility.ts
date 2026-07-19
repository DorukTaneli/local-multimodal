// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId } from "../chat/external-providers.ts";
import { isThreadIncognito } from "../chat/utils/thread-ids.ts";

export function isComfyGenerationEligible(args: {
  checkpoint: string;
  incognito: boolean;
  modelLoading: boolean;
  threadRunning: boolean;
  remoteId: string | undefined;
  activeThreadId: string | null;
  visibleText: string;
}): boolean {
  return Boolean(
    args.checkpoint &&
      !isExternalModelId(args.checkpoint) &&
      !args.incognito &&
      !args.modelLoading &&
      !args.threadRunning &&
      args.remoteId &&
      !isThreadIncognito(args.remoteId) &&
      args.remoteId === args.activeThreadId &&
      args.visibleText.trim(),
  );
}
