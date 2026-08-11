[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$IsaacPython,

    [Parameter(Mandatory = $true)]
    [string]$Runner,

    [Parameter(Mandatory = $true)]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$ProgressPath,

    [Parameter(Mandatory = $true)]
    [string]$StdoutPath,

    [Parameter(Mandatory = $true)]
    [string]$StderrPath,

    [Parameter(Mandatory = $true)]
    [string]$StatusPath,

    [string]$VisibleCamera = "CAM_11",

    [double]$SecondsPerDay = 60.0
)

$ErrorActionPreference = "Stop"

foreach ($requiredPath in @($IsaacPython, $Runner, $Stage)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required FireViewer launch input does not exist: $requiredPath"
    }
}

$logDirectory = Split-Path -Parent $StdoutPath
$statusDirectory = Split-Path -Parent $StatusPath
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $statusDirectory -Force | Out-Null

$launchArguments = @(
    $Runner,
    "--stage", $Stage,
    "--simulation-only",
    "--seconds-per-day", $SecondsPerDay.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--progress-path", $ProgressPath,
    "--visible-camera", $VisibleCamera
)

$startedAt = (Get-Date).ToUniversalTime().ToString("o")
$kitProcess = Start-Process `
    -FilePath $IsaacPython `
    -ArgumentList $launchArguments `
    -WorkingDirectory (Split-Path -Parent $Runner) `
    -WindowStyle Normal `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -PassThru

[ordered]@{
    status = "running"
    launcher_pid = $PID
    kit_process_id = $kitProcess.Id
    started_at_utc = $startedAt
    stage = $Stage
    visible_camera = $VisibleCamera
    seconds_per_day = $SecondsPerDay
    capture_enabled = $false
} | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8

$kitProcess.WaitForExit()
$exitCode = $kitProcess.ExitCode

[ordered]@{
    status = "exited"
    launcher_pid = $PID
    kit_process_id = $kitProcess.Id
    started_at_utc = $startedAt
    ended_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    exit_code = $exitCode
    stage = $Stage
    visible_camera = $VisibleCamera
    seconds_per_day = $SecondsPerDay
    capture_enabled = $false
} | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8

exit $exitCode
