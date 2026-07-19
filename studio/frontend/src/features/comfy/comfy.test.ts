// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// <reference types="node" />

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
});

test("initial generation orders preflight, eject, generate, then persistence", async () => {
  const calls: string[] = [];
  await runGenerationAfterReview({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => calls.push("ready"),
    eject: async () => {
      calls.push("eject");
      return true;
    },
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persist: async () => {
      calls.push("persist");
    },
  });
  assert.deepEqual(calls, ["status", "ready", "eject", "generate", "persist"]);
});

test("initial failures do not persist incomplete tool parts", async () => {
  let persisted = false;
  await assert.rejects(
    runGenerationAfterReview({
      status: async () => ready,
      requireReady: () => undefined,
      eject: async () => true,
      generate: async () => {
        throw new Error("generation failed");
      },
      persist: async () => {
        persisted = true;
      },
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
      eject: async () => {
        calls.push("eject");
        return true;
      },
      generate: async () => {
        calls.push("generate");
        return generated;
      },
      persist: async () => {
        calls.push("persist");
      },
    }),
  );
  assert.deepEqual(calls, ["status", "ready"]);
});

test("reroll skips eject without a local model and persists only on success", async () => {
  const calls: string[] = [];
  await runComfyReroll({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => undefined,
    shouldEject: () => false,
    eject: async () => {
      calls.push("eject");
      return true;
    },
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persistAsset: async () => {
      calls.push("persist");
    },
  });
  assert.deepEqual(calls, ["status", "generate", "persist"]);
});

test("failed rerolls leave the persisted tool part unchanged", async () => {
  let persisted = false;
  await assert.rejects(
    runComfyReroll({
      status: async () => ready,
      requireReady: () => undefined,
      shouldEject: () => true,
      eject: async () => true,
      generate: async () => {
        throw new Error("generation failed");
      },
      persistAsset: async () => {
        persisted = true;
      },
    }),
  );
  assert.equal(persisted, false);
});

test("reroll with a loaded local model ejects before generation", async () => {
  const calls: string[] = [];
  await runComfyReroll({
    status: async () => {
      calls.push("status");
      return ready;
    },
    requireReady: () => calls.push("ready"),
    shouldEject: () => true,
    eject: async () => {
      calls.push("eject");
      return true;
    },
    generate: async () => {
      calls.push("generate");
      return generated;
    },
    persistAsset: async () => {
      calls.push("persist");
    },
  });
  assert.deepEqual(calls, ["status", "ready", "eject", "generate", "persist"]);
});

test("central local-model load releases ComfyUI before unload and load", () => {
  const source = readFileSync(
    new URL("../chat/hooks/use-chat-model-runtime.ts", import.meta.url),
    "utf8",
  );
  const releaseIndex = source.indexOf("await releaseComfyBeforeLocalModelLoad()");
  const unloadIndex = source.indexOf("await unloadModel({ model_path: currentCheckpoint })");
  const loadIndex = source.indexOf("const loadResponse = await loadModel({");
  assert.ok(releaseIndex >= 0, "central model-load seam must release ComfyUI");
  assert.ok(releaseIndex < unloadIndex, "ComfyUI release must precede local chat unload");
  assert.ok(releaseIndex < loadIndex, "ComfyUI release must precede local chat load");
});
