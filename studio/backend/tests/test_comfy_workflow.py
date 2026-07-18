# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from features.comfy import workflow as workflow_module
from features.comfy.workflow import (
    WORKFLOW_FILENAME,
    WorkflowError,
    effective_prompt,
    mutate_workflow,
    resolve_workflow_path,
    validate_workflow,
)


def _canonical() -> dict:
    path = workflow_module.REPOSITORY_ROOT / "comfy" / WORKFLOW_FILENAME
    return json.loads(path.read_text(encoding = "utf-8"))


def _changed_paths(before, after, prefix = ()) -> set[tuple[str, ...]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[tuple[str, ...]] = set()
        for key in before.keys() | after.keys():
            changed |= _changed_paths(before.get(key), after.get(key), (*prefix, str(key)))
        return changed
    return {prefix} if before != after else set()


def test_mutation_is_immutable_and_changes_only_required_inputs():
    canonical = _canonical()
    original = deepcopy(canonical)

    mutated = mutate_workflow(
        canonical,
        prompt = "masterpiece, 1girl, wink",
        seed = 42,
        filename_prefix = "LocalMultimodal_test",
    )

    assert canonical == original
    assert mutated is not canonical
    assert _changed_paths(canonical, mutated) == {
        ("2", "inputs", "text"),
        ("5", "inputs", "seed"),
        ("7", "inputs", "filename_prefix"),
    }


def test_effective_prompt_uses_defaults_and_ignores_blank_sections():
    canonical = _canonical()
    assert effective_prompt(
        canonical, image_tags = "  1girl, wink  ", conversation_tags = "   "
    ) == "masterpiece, best quality, amazing quality, absurdres, 1girl, wink"


def test_validation_reports_missing_required_node():
    workflow = _canonical()
    del workflow["7"]
    with pytest.raises(WorkflowError, match = "missing required node 7"):
        validate_workflow(workflow)


def test_resolution_prefers_installed_workflow(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    installed = studio / "share" / "workflows" / WORKFLOW_FILENAME
    installed.parent.mkdir(parents = True)
    installed.write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(workflow_module, "studio_root", lambda: studio)

    assert resolve_workflow_path() == installed


def test_resolution_reports_all_missing_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_module, "studio_root", lambda: tmp_path / "studio")
    monkeypatch.setattr(workflow_module, "REPOSITORY_ROOT", tmp_path / "repository")
    with pytest.raises(WorkflowError, match = "workflow is missing"):
        resolve_workflow_path()
