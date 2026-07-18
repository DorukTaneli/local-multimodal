# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve, validate, and immutably specialize the canonical ComfyUI workflow."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from utils.paths.storage_roots import studio_root


WORKFLOW_FILENAME = "prefect_illustrious_xl_basic_comfyui_workflow.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

_REQUIRED_NODES = {
    "1": "CheckpointLoaderSimple",
    "2": "CLIPTextEncode",
    "3": "CLIPTextEncode",
    "4": "EmptyLatentImage",
    "5": "KSampler",
    "6": "VAEDecode",
    "7": "SaveImage",
}


class WorkflowError(RuntimeError):
    """The canonical workflow is missing or invalid."""


def workflow_candidates() -> tuple[Path, Path]:
    return (
        studio_root() / "share" / "workflows" / WORKFLOW_FILENAME,
        REPOSITORY_ROOT / "comfy" / WORKFLOW_FILENAME,
    )


def resolve_workflow_path() -> Path:
    candidates = workflow_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise WorkflowError(f"ComfyUI workflow is missing. Expected one of: {searched}")


def load_workflow() -> dict[str, Any]:
    path = resolve_workflow_path()
    try:
        raw = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Could not load ComfyUI workflow at {path}: {exc}") from exc
    validate_workflow(raw)
    return raw


def _node_inputs(workflow: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise WorkflowError(f"ComfyUI workflow is missing required node {node_id}.")
    expected_type = _REQUIRED_NODES[node_id]
    if node.get("class_type") != expected_type:
        raise WorkflowError(
            f"ComfyUI workflow node {node_id} must be {expected_type}, "
            f"not {node.get('class_type')!r}."
        )
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise WorkflowError(f"ComfyUI workflow node {node_id} has no valid inputs.")
    return inputs


def validate_workflow(workflow: Any) -> None:
    if not isinstance(workflow, dict):
        raise WorkflowError("ComfyUI workflow must be a JSON object.")
    for node_id in _REQUIRED_NODES:
        _node_inputs(workflow, node_id)

    checkpoint = _node_inputs(workflow, "1").get("ckpt_name")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise WorkflowError("ComfyUI workflow node 1 has no checkpoint name.")
    prompt = _node_inputs(workflow, "2").get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkflowError("ComfyUI workflow node 2 has no positive prompt defaults.")
    sampler = _node_inputs(workflow, "5")
    if not isinstance(sampler.get("seed"), int):
        raise WorkflowError("ComfyUI workflow node 5 has no integer seed.")
    save = _node_inputs(workflow, "7")
    if not isinstance(save.get("filename_prefix"), str):
        raise WorkflowError("ComfyUI workflow node 7 has no filename prefix.")
    latent = _node_inputs(workflow, "4")
    for name in ("width", "height"):
        value = latent.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise WorkflowError(f"ComfyUI workflow node 4 has no valid {name}.")


def checkpoint_name(workflow: dict[str, Any]) -> str:
    validate_workflow(workflow)
    return str(workflow["1"]["inputs"]["ckpt_name"])


def image_dimensions(workflow: dict[str, Any]) -> tuple[int, int]:
    validate_workflow(workflow)
    inputs = workflow["4"]["inputs"]
    return int(inputs["width"]), int(inputs["height"])


def effective_prompt(
    workflow: dict[str, Any],
    *,
    image_tags: str,
    conversation_tags: str,
) -> str:
    validate_workflow(workflow)
    defaults = workflow["2"]["inputs"]["text"]
    return ", ".join(
        section.strip() for section in (defaults, conversation_tags, image_tags) if section.strip()
    )


def mutate_workflow(
    workflow: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    """Return a deep copy with only the three per-generation inputs changed."""
    validate_workflow(workflow)
    if not prompt.strip():
        raise ValueError("The effective prompt cannot be blank.")
    if seed < 0:
        raise ValueError("The seed cannot be negative.")
    if not filename_prefix.strip():
        raise ValueError("The filename prefix cannot be blank.")

    mutated = deepcopy(workflow)
    mutated["2"]["inputs"]["text"] = prompt
    mutated["5"]["inputs"]["seed"] = seed
    mutated["7"]["inputs"]["filename_prefix"] = filename_prefix
    return mutated
