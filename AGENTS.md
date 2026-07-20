# Local Multimodal Development Rules

## Preserve easy upstream synchronization

This repository is a fork of Unsloth. Keeping future merges and rebases from
upstream simple is a primary architectural requirement.

- Put fork-specific functionality in new, isolated modules whenever possible.
- Limit edits to upstream-owned files to stable integration seams: imports,
  route or feature registration, and small lifecycle hooks.
- Do not copy, rewrite, or broadly refactor upstream inference, runtime, or UI
  internals to implement fork features.
- Keep accelerator coordination and ComfyUI behavior under the fork-owned
  `studio/backend/features/comfy/` boundary. Expose a small API to upstream
  inference code instead of spreading ownership or state across layers.
- Prefer backend-owned, atomic resource coordination over frontend sequences of
  independent requests. Frontend code should request an operation, not manage
  GPU lifecycle transitions itself.
- Before changing an existing upstream file, consider whether the same behavior
  can be implemented in a new fork-owned file with a narrower hook.
- When a change to an upstream file is unavoidable, keep it minimal and document
  why the integration seam is required.

Treat the size and location of the diff against upstream as part of correctness,
not merely as a maintenance preference.
