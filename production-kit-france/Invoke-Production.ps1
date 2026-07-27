[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [ValidateSet('Plan', 'Execute')]
    [string]$Mode = 'Plan',

    [ValidateSet(
        'all',
        'plan_05m',
        'produce_05m',
        'near_imagery',
        'far_rasters',
        'far_imagery',
        'vector_package',
        'blender_scene',
        'unity_catalog',
        'validate_catalog',
        'site_upload'
    )]
    [string]$Stage = 'all',

    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_production.py'
$arguments = @($runner, '--config', (Resolve-Path -LiteralPath $Config).Path)
if ($Mode -eq 'Plan') {
    $arguments += '--plan'
}
else {
    $arguments += @('--execute', '--stage', $Stage)
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "La production FireViewer a echoue avec le code $LASTEXITCODE."
}
