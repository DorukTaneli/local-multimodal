// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// <reference types="node" />

import assert from "node:assert/strict";
import test from "node:test";
import { markThreadIncognito } from "../chat/utils/thread-ids.ts";
import { isComfyGenerationEligible } from "./eligibility.ts";
import { runComfyReroll, runGenerationAfterReview } from "./operations.ts";
import {
  buildContextProjection,
  extractComfyContextProjection,
} from "./serialization.ts";
import type { ComfyStatus } from "./types.ts";

const ready: ComfyStatus = {
  ready: true,
  reachable: true,
  workflowValid: true,
  checkpointAvailable: true,
  busy: false,
  checkpoint: "checkpoint.safetensors",
  failures: [],
};

const generated = {
  effectivePrompt: "quality, scene",
  asset: {
    assetId: "asset-1",
    seed: 7,
    width: 896,
    height: 1152,
    mimeType: "image/png" as const,
  },
};

test("persisted assistant-ui local ids are eligible but incognito threads stay protected", () => {
  // Persisted threads reload without an in-memory incognito marker even though
  // assistant-ui keeps the client-generated `__LOCALID_` as their remote id.
  const persistedThreadId = "__LOCALID_persisted";
  const temporaryThreadId = "__LOCALID_incognito";
  markThreadIncognito(temporaryThreadId);

  const eligible = (remoteId: string) =>
    isComfyGenerationEligible({
      checkpoint: "local-model.gguf",
      incognito: false,
      modelLoading: false,
      threadRunning: false,
      remoteId,
      activeThreadId: remoteId,
      visibleText: "A scene worth illustrating",
    });

  assert.equal(eligible(persistedThreadId), true);
  assert.equal(eligible(temporaryThreadId), false);
});

test("Comfy projection includes reviewed tags but not workflow defaults", () => {
  assert.equal(
    buildContextProjection("night, rain", "1girl, wink"),
    "[Image: night, rain, 1girl, wink]",
  );
});

test("serializer accepts only a complete Comfy result", () => {
  const complete = {
    type: "tool-call",
    toolName: "comfy_generate_image",
    result: {
      effectivePrompt: "quality, 1girl",
      assets: [generated.asset],
      contextProjection: "[Image: 1girl]",
    },
  };
  assert.equal(extractComfyContextProjection(complete), "[Image: 1girl]");
  assert.equal(
    extractComfyContextProjection({ ...complete, result: { ...complete.result, assets: [] } }),
    null,
  );
  assert.equal(
    extractComfyContextProjection({ ...complete, toolName: "image_generation" }),
    null,
  );
  assert.equal(
    extractComfyContextProjection({
      ...complete,
      result: {
        ...complete.result,
        assets: [{ ...generated.asset, seed: Number.MAX_SAFE_INTEGER + 1 }],
      },
    }),
    null,
  );
});

test("initial generation orders preflight, generate, then persistence", async () => {
  const calls: string[] = [];
  await runGenerationAfterReview({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => calls.push("ready"),
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persist: async () => {
      calls.push("persist");
    },
    discard: async () => calls.push("discard"),
  });
  assert.deepEqual(calls, ["status", "ready", "generate", "persist"]);
});

test("initial failures do not persist incomplete tool parts", async () => {
  let persisted = false;
  await assert.rejects(
    runGenerationAfterReview({
      status: async () => ready,
      requireReady: () => undefined,
      generate: async () => {
        throw new Error("generation failed");
      },
      persist: async () => {
        persisted = true;
      },
      discard: async () => undefined,
    }),
  );
  assert.equal(persisted, false);
});

test("failed preflight leaves the chat model loaded", async () => {
  const calls: string[] = [];
  await assert.rejects(
    runGenerationAfterReview({
      status: async () => {
        calls.push("status");
        return { ...ready, ready: false, failures: ["checkpoint missing"] };
      },
      requireReady: () => {
        calls.push("ready");
        throw new Error("checkpoint missing");
      },
      generate: async () => {
        calls.push("generate");
        return generated;
      },
      persist: async () => {
        calls.push("persist");
      },
      discard: async () => calls.push("discard"),
    }),
  );
  assert.deepEqual(calls, ["status", "ready"]);
});

test("reroll persists only after backend generation succeeds", async () => {
  const calls: string[] = [];
  await runComfyReroll({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => undefined,
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persistAsset: async () => {
      calls.push("persist");
    },
    discard: async () => calls.push("discard"),
  });
  assert.deepEqual(calls, ["status", "generate", "persist"]);
});

test("failed rerolls leave the persisted tool part unchanged", async () => {
  let persisted = false;
  await assert.rejects(
    runComfyReroll({
      status: async () => ready,
      requireReady: () => undefined,
      generate: async () => {
        throw new Error("generation failed");
      },
      persistAsset: async () => {
        persisted = true;
      },
      discard: async () => undefined,
    }),
  );
  assert.equal(persisted, false);
});

test("failed reroll persistence discards the unreferenced generated asset", async () => {
  const calls: string[] = [];
  await runComfyReroll({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => calls.push("ready"),
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persistAsset: async () => {
      calls.push("persist");
      throw new Error("save failed");
    },
    discard: async () => calls.push("discard"),
  }).then(
    () => assert.fail("persistence failure must reject"),
    () => undefined,
  );
  assert.deepEqual(calls, ["status", "ready", "generate", "persist", "discard"]);
});
