# Local Multimodal installer for Windows PowerShell.
# Run this script from a Local Multimodal clone.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    throw "Local Multimodal installation requires USERPROFILE to be set."
}

$env:UNSLOTH_STUDIO_HOME = "$env:USERPROFILE\.local-multimodal\studio"
$env:UNSLOTH_SKIP_AUTOSTART = "1"

$workflowName = "prefect_illustrious_xl_basic_comfyui_workflow.json"
$workflowSource = Join-Path $PSScriptRoot "comfy\$workflowName"
if (-not (Test-Path -LiteralPath $workflowSource -PathType Leaf)) {
    throw "Local Multimodal workflow is missing: $workflowSource"
}

try {
    $LASTEXITCODE = 0
    & (Join-Path $PSScriptRoot "install.ps1") --local
    if ($LASTEXITCODE -ne 0) {
        throw "install.ps1 exited with code $LASTEXITCODE."
    }
} catch {
    throw "Local Multimodal installation stopped because the base installer failed: $($_.Exception.Message)"
}

$studioRoot = $env:UNSLOTH_STUDIO_HOME
$workflowDirectory = Join-Path $studioRoot "share\workflows"
New-Item -ItemType Directory -Path $workflowDirectory -Force | Out-Null
Copy-Item -LiteralPath $workflowSource -Destination (Join-Path $workflowDirectory $workflowName) -Force

$privateUnslothExe = Join-Path $studioRoot "unsloth_studio\Scripts\unsloth.exe"
if (-not (Test-Path -LiteralPath $privateUnslothExe -PathType Leaf)) {
    throw "Private Local Multimodal CLI was not found after installation: $privateUnslothExe"
}

$binDirectory = Join-Path $env:USERPROFILE ".local-multimodal\bin"
$wrapperPath = Join-Path $binDirectory "local-multimodal.cmd"
New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null
$wrapperContent = @'
@echo off
set "UNSLOTH_STUDIO_HOME=%USERPROFILE%\.local-multimodal\studio"
"%USERPROFILE%\.local-multimodal\studio\unsloth_studio\Scripts\unsloth.exe" studio %*
exit /b %ERRORLEVEL%
'@
[System.IO.File]::WriteAllText($wrapperPath, $wrapperContent, [System.Text.Encoding]::ASCII)

$userPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)
$normalizedBinDirectory = $binDirectory.TrimEnd('\')
$hasBinDirectory = $false
if (-not [string]::IsNullOrWhiteSpace($userPath)) {
    foreach ($entry in $userPath.Split(';')) {
        $normalizedEntry = [Environment]::ExpandEnvironmentVariables($entry.Trim()).TrimEnd('\')
        if ($normalizedEntry -ieq $normalizedBinDirectory) {
            $hasBinDirectory = $true
            break
        }
    }
}

if (-not $hasBinDirectory) {
    $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
        $binDirectory
    } else {
        "$($userPath.TrimEnd(';'));$binDirectory"
    }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, [EnvironmentVariableTarget]::User)
}

Write-Host ""
Write-Host "Local Multimodal installation complete."
Write-Host "Open a new PowerShell window, then launch with:"
Write-Host "  local-multimodal -p 8889"
