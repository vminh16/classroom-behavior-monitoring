[CmdletBinding()]
param(
    [ValidateSet("video", "camera", "rtsp")]
    [string]$SourceType = "video",

    [string]$VideoFile = "",
    [int]$CameraId = 0,
    [string]$RtspUrl = "",
    [string]$OutputDir = "",
    [string]$EnvName = "paddle_det",
    [string]$Config = "",
    [string]$SecretsFile = "",
    [string]$RunMode = "paddle",
    [string]$CondaExe = "",
    [switch]$UseTelegram,
    [switch]$Cpu,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CondaExe {
    param(
        [string]$ExplicitPath
    )

    if ($ExplicitPath -and (Test-Path $ExplicitPath)) {
        return (Resolve-Path $ExplicitPath).Path
    }

    $candidates = @()
    if ($env:CONDA_EXE) {
        $candidates += $env:CONDA_EXE
    }
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
        $candidates += (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe")
    }
    $candidates += "C:\\Users\\USER\\anaconda3\\Scripts\\conda.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Cannot find conda.exe. Pass -CondaExe with the full path to conda.exe."
}

function Load-DotEnv {
    param(
        [string]$Path
    )

    $values = @{}
    if (-not $Path -or -not (Test-Path $Path)) {
        return $values
    }

    foreach ($line in Get-Content -Path $Path) {
        $text = $line.Trim()
        if (-not $text -or $text.StartsWith("#")) {
            continue
        }
        $pair = $text -split "=", 2
        if ($pair.Count -ne 2) {
            continue
        }
        $key = $pair[0].Trim()
        $value = $pair[1].Trim()
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        $values[$key] = $value
    }
    return $values
}

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $BundleRoot

if (-not $Config) {
    $Config = Join-Path $BundleRoot "config\demo_final.yml"
}
if (-not $SecretsFile) {
    $SecretsFile = Join-Path $BundleRoot "secrets\telegram.env"
}
if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ProjectRoot ("output\demo_" + $stamp)
}
if (-not $VideoFile) {
    $VideoFile = Join-Path $ProjectRoot "test_data\data.mp4"
}

$CondaExe = Resolve-CondaExe -ExplicitPath $CondaExe
$Secrets = Load-DotEnv -Path $SecretsFile

$RequiredPaths = @(
    (Join-Path $ProjectRoot "output_inference\picodet_m_416_classroom"),
    (Join-Path $ProjectRoot "output_inference\pplcnet_behavior"),
    $Config
)
foreach ($required in $RequiredPaths) {
    if (-not (Test-Path $required)) {
        throw "Required path not found: $required. Restore runtime artifacts first; see docs/ARTIFACTS.md."
    }
}

$device = if ($Cpu) { "CPU" } else { "GPU" }

$optArgs = @(
    "visual=True",
    "visual_style=minimal"
)

if ($UseTelegram) {
    $botToken = $Secrets["TELEGRAM_BOT_TOKEN"]
    $chatId = $Secrets["TELEGRAM_CHAT_ID"]
    if (-not $botToken -or -not $chatId) {
        throw "Telegram was requested but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in $SecretsFile"
    }
    $optArgs += "TELEGRAM_ALERT.enable=True"
    $optArgs += "TELEGRAM_ALERT.bot_token=$botToken"
    $optArgs += "TELEGRAM_ALERT.chat_id=$chatId"
}
else {
    $optArgs += "TELEGRAM_ALERT.enable=False"
}

$command = @(
    $CondaExe,
    "run",
    "-n",
    $EnvName,
    "python",
    "deploy/pipeline/pipeline_product.py",
    "--config",
    $Config,
    "--device",
    $device,
    "--run_mode",
    $RunMode,
    "--output_dir",
    $OutputDir
)

switch ($SourceType) {
    "video" {
        if (-not (Test-Path $VideoFile)) {
            throw "Video file not found: $VideoFile. Restore sample inputs first; see docs/ARTIFACTS.md."
        }
        $command += @("--video_file", $VideoFile)
    }
    "camera" {
        $command += @("--camera_id", $CameraId.ToString())
    }
    "rtsp" {
        if (-not $RtspUrl) {
            throw "SourceType=rtsp requires -RtspUrl"
        }
        $command += @("--rtsp", $RtspUrl)
    }
}

if ($optArgs.Count -gt 0) {
    $command += "--opt"
    $command += $optArgs
}

Write-Host "=== Demo Deploy Bundle ==="
Write-Host ("Project root : " + $ProjectRoot)
Write-Host ("Config       : " + $Config)
Write-Host ("Environment  : " + $EnvName)
Write-Host ("Conda        : " + $CondaExe)
Write-Host ("Device       : " + $device)
Write-Host ("Run mode     : " + $RunMode)
Write-Host ("Source type  : " + $SourceType)
if ($SourceType -eq "video") {
    Write-Host ("Video file   : " + $VideoFile)
}
elseif ($SourceType -eq "camera") {
    Write-Host ("Camera ID    : " + $CameraId)
}
else {
    Write-Host ("RTSP URL     : " + $RtspUrl)
}
Write-Host ("Output dir   : " + $OutputDir)
Write-Host ("Telegram     : " + ($(if ($UseTelegram) { "enabled via env file" } else { "disabled" })))
Write-Host ""
Write-Host "Command:"
Write-Host (($command | ForEach-Object {
            if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
        }) -join " ")

if ($DryRun) {
    return
}

Push-Location $ProjectRoot
try {
    & $command[0] $command[1..($command.Count - 1)]
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
