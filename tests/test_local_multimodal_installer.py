"""Static and parser checks for the isolated Local Multimodal Windows installer."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "install-local-multimodal.ps1"
INDEX_HTML = REPO_ROOT / "studio" / "frontend" / "index.html"
README = REPO_ROOT / "README.md"


def _source() -> str:
    return INSTALLER.read_text(encoding = "utf-8")


@pytest.mark.skipif(
    shutil.which("pwsh") is None and shutil.which("powershell") is None,
    reason = "PowerShell is unavailable",
)
def test_installer_has_valid_powershell_syntax():
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    path = str(INSTALLER).replace("'", "''")
    parser_check = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', "
        "[ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", parser_check],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_invokes_relative_base_installer_and_stops_on_failure():
    source = _source()
    invocation = '& (Join-Path $PSScriptRoot "install.ps1") --local'
    assert source.count(invocation) == 1
    assert re.search(
        re.escape(invocation)
        + r'.*?if \(\$LASTEXITCODE -ne 0\).*?throw "install\.ps1 exited with code',
        source,
        flags = re.DOTALL,
    )
    assert "the base installer failed" in source


def test_uses_only_the_private_local_multimodal_home():
    source = _source()
    assert (
        '$env:UNSLOTH_STUDIO_HOME = "$env:USERPROFILE\\.local-multimodal\\studio"'
        in source
    )
    assert '$env:UNSLOTH_SKIP_AUTOSTART = "1"' in source
    assert "%USERPROFILE%\\.unsloth" not in source
    assert 'Join-Path $env:USERPROFILE ".unsloth' not in source


def test_wrapper_targets_private_cli_and_forwards_arguments():
    source = _source()
    assert (
        '"%USERPROFILE%\\.local-multimodal\\studio\\unsloth_studio\\Scripts\\unsloth.exe" '
        "studio %*"
        in source
    )
    assert (
        'set "UNSLOTH_STUDIO_HOME=%USERPROFILE%\\.local-multimodal\\studio"'
        in source
    )
    assert 'exit /b %ERRORLEVEL%' in source
    assert 'Join-Path $studioRoot "unsloth_studio\\Scripts\\unsloth.exe"' in source


def test_copies_canonical_workflow_after_base_install():
    source = _source()
    workflow = "prefect_illustrious_xl_basic_comfyui_workflow.json"
    assert f'$workflowName = "{workflow}"' in source
    assert 'Join-Path $PSScriptRoot "comfy\\$workflowName"' in source
    assert 'Join-Path $studioRoot "share\\workflows"' in source
    assert source.index("Copy-Item -LiteralPath $workflowSource") > source.index(
        '& (Join-Path $PSScriptRoot "install.ps1") --local'
    )


def test_user_path_update_is_case_insensitive_and_idempotent():
    source = _source()
    assert 'Join-Path $env:USERPROFILE ".local-multimodal\\bin"' in source
    assert '[Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)' in source
    assert "$normalizedEntry -ieq $normalizedBinDirectory" in source
    assert "if (-not $hasBinDirectory)" in source
    assert (
        '[Environment]::SetEnvironmentVariable("Path", $newUserPath, '
        "[EnvironmentVariableTarget]::User)"
        in source
    )
    path_update = source[source.index("$userPath ="):]
    assert "Scripts\\unsloth.exe" not in path_update


def test_visible_identity_and_coexistence_documentation_are_narrow():
    identity = "Local Multimodal, based on Unsloth Studio"
    assert f"<title>{identity}</title>" in INDEX_HTML.read_text(encoding = "utf-8")
    readme = README.read_text(encoding = "utf-8")
    assert ".\\install-local-multimodal.ps1" in readme
    assert "local-multimodal -p 8889" in readme
    assert "prefectIllustriousXL_v70.safetensors" in readme
