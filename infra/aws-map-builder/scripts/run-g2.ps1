[CmdletBinding()]
param(
    [string]$Profile = 'unicorn-whodev',
    [string]$Region = 'eu-west-3',
    [string]$AwsPath = 'C:\Users\charl\AppData\Local\Programs\Amazon\AWSCLIV2\aws.exe',
    [string]$ExpectedImageDigest = 'sha256:64b70a0e227e68336126bf6833b2b417f34a0556685a4caf93c88d9181ae5ecf',
    [string]$ExpectedGitCommit = '766f157d00e15da72271ec197706c203f040fb7a',
    [int]$MaximumRuntimeMinutes = 120
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$env:AWS_CLI_OUTPUT_ENCODING = 'utf-8'

function Invoke-AwsJson {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $raw = & $AwsPath @Arguments --profile $Profile --output json
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed: $($Arguments -join ' ')"
    }
    if ([string]::IsNullOrWhiteSpace(($raw -join "`n"))) {
        return $null
    }
    return (($raw -join "`n") | ConvertFrom-Json -Depth 100)
}

if (-not (Test-Path -LiteralPath $AwsPath -PathType Leaf)) {
    throw "AWS CLI was not found: $AwsPath"
}
if ($Region -ne 'eu-west-3') {
    throw 'G2 is locked to eu-west-3'
}
if ($MaximumRuntimeMinutes -gt 120 -or $MaximumRuntimeMinutes -lt 15) {
    throw 'MaximumRuntimeMinutes must remain between 15 and 120'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$requestPath = Join-Path $repoRoot 'reference\map-builder-reference-v1\request.json'
$baselinePath = Join-Path $repoRoot 'reference\map-builder-reference-v1\semantic-baseline.json'
$remoteScriptPath = Join-Path $PSScriptRoot 'execute-g2.sh'
$comparatorPath = Join-Path $repoRoot 'runtime\compare_semantic_parity.py'

$request = Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json -Depth 50
if ($request.schema -ne 'fireviewer.map-job.v1' -or $request.builder.git_commit -ne $ExpectedGitCommit -or $request.builder.image_digest -ne $ExpectedImageDigest) {
    throw 'The golden request does not match the locked builder identity'
}

$identity = Invoke-AwsJson sts get-caller-identity --region $Region
$accountId = $identity.Account
$workBucket = "fireviewer-map-work-$accountId-$Region"
$buildsBucket = "fireviewer-map-builds-$accountId-$Region"
$repository = "$accountId.dkr.ecr.$Region.amazonaws.com/fireviewer-map-builder"
$imageRef = "$repository@$ExpectedImageDigest"
$outputPrefix = "maps/$($request.zone_id)/$($request.build_id)"
$requestKey = "requests/$($request.build_id)/request.json"
$baselineKey = "requests/$($request.build_id)/semantic-baseline.json"
$comparatorKey = "requests/$($request.build_id)/compare-semantic-parity.py"
$remoteScriptKey = "requests/$($request.build_id)/execute-g2.sh"

$launchTemplates = Invoke-AwsJson ec2 describe-launch-templates --region $Region --filters 'Name=tag:Application,Values=FireViewer' 'Name=tag:Component,Values=MapBuilder'
$launchTemplate = @($launchTemplates.LaunchTemplates | Where-Object { $_.LaunchTemplateName -like 'fireviewer-map-builder-g2-*' })
if ($launchTemplate.Count -ne 1) {
    throw "Expected exactly one G2 launch template; found $($launchTemplate.Count)"
}
$launchTemplateId = $launchTemplate[0].LaunchTemplateId
$launchTemplateVersion = [string]$launchTemplate[0].LatestVersionNumber

$evidenceRoot = Join-Path (Split-Path -Parent $repoRoot) ".codex-temp\aws-map-builder-g2-$($request.build_id)"
New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null
$preflightPath = Join-Path $evidenceRoot 'preflight.json'

& (Join-Path $PSScriptRoot 'preflight-g2.ps1') `
    -AwsPath $AwsPath `
    -Profile $Profile `
    -Region $Region `
    -ExpectedAccountId $accountId `
    -ExpectedImageDigest $ExpectedImageDigest `
    -LaunchTemplateId $launchTemplateId `
    -BuildsBucket $buildsBucket `
    -OutputPrefix $outputPrefix `
    -EvidencePath $preflightPath | Out-Host

& $AwsPath s3 cp $requestPath "s3://$workBucket/$requestKey" --profile $Profile --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'request.json upload failed' }
& $AwsPath s3 cp $baselinePath "s3://$workBucket/$baselineKey" --profile $Profile --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'semantic baseline upload failed' }
& $AwsPath s3 cp $comparatorPath "s3://$workBucket/$comparatorKey" --profile $Profile --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'semantic comparator upload failed' }
& $AwsPath s3 cp $remoteScriptPath "s3://$workBucket/$remoteScriptKey" --profile $Profile --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'G2 execution script upload failed' }

$instanceId = $null
$commandId = $null
$failure = $null
$startedAt = [datetimeoffset]::UtcNow
try {
    $clientToken = "fireviewer-g2-$([guid]::NewGuid().ToString('N'))"
    $run = Invoke-AwsJson ec2 run-instances --region $Region --launch-template "LaunchTemplateId=$launchTemplateId,Version=$launchTemplateVersion" --count 1 --client-token $clientToken
    $instanceId = $run.Instances[0].InstanceId
    if ([string]::IsNullOrWhiteSpace($instanceId)) {
        throw 'EC2 did not return an instance id'
    }
    Write-Host "G2 instance launched: $instanceId"

    & $AwsPath ec2 wait instance-running --profile $Profile --region $Region --instance-ids $instanceId
    if ($LASTEXITCODE -ne 0) { throw 'EC2 did not reach running state' }

    $ssmOnline = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        $information = Invoke-AwsJson ssm describe-instance-information --region $Region --filters "Key=InstanceIds,Values=$instanceId"
        if (@($information.InstanceInformationList | Where-Object { $_.PingStatus -eq 'Online' }).Count -eq 1) {
            $ssmOnline = $true
            break
        }
        Start-Sleep -Seconds 10
    }
    if (-not $ssmOnline) { throw 'SSM did not become online within 15 minutes' }

    $remoteCommands = @(
        'set -e'
        'echo "Waiting for the FireViewer scratch bootstrap"'
        'for attempt in $(seq 1 180); do if [ -f /scratch/.fireviewer-ready ]; then break; fi; sleep 5; done'
        'if [ ! -f /scratch/.fireviewer-ready ]; then echo "FireViewer scratch bootstrap did not complete" >&2; exit 70; fi'
        'echo "FireViewer scratch bootstrap is ready"'
        'mkdir -p /scratch/fireviewer-control'
        "aws s3 cp s3://$workBucket/$remoteScriptKey /scratch/fireviewer-control/execute-g2.sh --only-show-errors"
        'chmod 0500 /scratch/fireviewer-control/execute-g2.sh'
        "/bin/bash /scratch/fireviewer-control/execute-g2.sh '$workBucket' '$requestKey' '$baselineKey' '$comparatorKey' '$buildsBucket' '$outputPrefix' '$imageRef' '$ExpectedImageDigest' '$ExpectedGitCommit' '$launchTemplateId' '$launchTemplateVersion' '$Region'"
    )
    $parameters = @{ commands = $remoteCommands } | ConvertTo-Json -Compress
    $cloudWatch = @{ CloudWatchLogGroupName = '/aws/ec2/fireviewer-map-builder'; CloudWatchOutputEnabled = $true } | ConvertTo-Json -Compress
    $sent = Invoke-AwsJson ssm send-command --region $Region --instance-ids $instanceId --document-name AWS-RunShellScript --comment "FireViewer G2 $($request.build_id)" --timeout-seconds 7200 --parameters $parameters --cloud-watch-output-config $cloudWatch
    $commandId = $sent.Command.CommandId
    if ([string]::IsNullOrWhiteSpace($commandId)) { throw 'SSM did not return a command id' }
    Write-Host "G2 command started: $commandId"

    $deadline = [datetimeoffset]::UtcNow.AddMinutes($MaximumRuntimeMinutes)
    $terminal = @('Success', 'Cancelled', 'TimedOut', 'Failed', 'Cancelling')
    do {
        Start-Sleep -Seconds 10
        $invocation = Invoke-AwsJson ssm get-command-invocation --region $Region --command-id $commandId --instance-id $instanceId
        Write-Host "G2 status: $($invocation.Status)"
        if ($terminal -contains $invocation.Status) { break }
    } while ([datetimeoffset]::UtcNow -lt $deadline)

    if ($invocation.Status -ne 'Success') {
        $failure = [ordered]@{
            status = $invocation.Status
            status_details = $invocation.StatusDetails
            response_code = $invocation.ResponseCode
            standard_error = $invocation.StandardErrorContent
            cloudwatch = $invocation.CloudWatchOutputConfig
        }
        throw "G2 command failed: $($invocation.Status) / $($invocation.StatusDetails)"
    }

    $done = Invoke-AwsJson s3api head-object --region $Region --checksum-mode ENABLED --bucket $buildsBucket --key "$outputPrefix/zone.done.json"
    if ($null -eq $done.Metadata.sha256 -or $null -eq $done.ChecksumSHA256) {
        throw 'zone.done.json exists without verified SHA-256 metadata'
    }

    $downloadFiles = @(
        'zone.done.json',
        'manifests/validation-result.json',
        'manifests/manifest.json',
        'manifests/hashes.json',
        'manifests/semantic-parity.json',
        'runtime/viewer-scene.v1.json',
        'metrics/build-metrics.json',
        'metrics/aws-execution-metrics.json',
        'provenance/aws-execution.json'
    )
    foreach ($relativePath in $downloadFiles) {
        $destination = Join-Path $evidenceRoot ($relativePath -replace '/', '\')
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        & $AwsPath s3 cp "s3://$buildsBucket/$outputPrefix/$relativePath" $destination --profile $Profile --region $Region --only-show-errors
        if ($LASTEXITCODE -ne 0) { throw "Evidence download failed: $relativePath" }
    }

    $localParityPath = Join-Path $evidenceRoot 'independent-semantic-parity.json'
    & (Get-Command python).Source $comparatorPath --baseline $baselinePath --output $evidenceRoot --write $localParityPath
    if ($LASTEXITCODE -ne 0) { throw 'Independent local semantic parity failed' }

    $receipt = [ordered]@{
        schema = 'fireviewer.aws-g2-run.v1'
        status = 'PASS'
        account_id = $accountId
        region = $Region
        instance_id = $instanceId
        command_id = $commandId
        launch_template_id = $launchTemplateId
        launch_template_version = $launchTemplateVersion
        image_digest = $ExpectedImageDigest
        git_commit = $ExpectedGitCommit
        output = "s3://$buildsBucket/$outputPrefix"
        started_at = $startedAt.ToString('o')
        completed_at = [datetimeoffset]::UtcNow.ToString('o')
    }
    $receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'g2-run.json') -Encoding utf8NoBOM
    $receipt | ConvertTo-Json -Depth 20 | Write-Host
}
catch {
    $failureReceipt = [ordered]@{
        schema = 'fireviewer.aws-g2-run.v1'
        status = 'FAIL'
        account_id = $accountId
        region = $Region
        instance_id = $instanceId
        command_id = $commandId
        error = $_.Exception.Message
        invocation = $failure
        started_at = $startedAt.ToString('o')
        failed_at = [datetimeoffset]::UtcNow.ToString('o')
    }
    $failurePath = Join-Path $evidenceRoot 'g2-failure.json'
    $failureReceipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $failurePath -Encoding utf8NoBOM
    try {
        & $AwsPath s3 cp $failurePath "s3://$workBucket/failed/$($request.build_id)/g2-failure.json" --profile $Profile --region $Region --only-show-errors | Out-Null
    }
    catch {
        Write-Warning "Failure receipt could not be uploaded: $($_.Exception.Message)"
    }
    throw
}
finally {
    if (-not [string]::IsNullOrWhiteSpace($instanceId)) {
        Write-Host "Terminating G2 instance: $instanceId"
        & $AwsPath ec2 terminate-instances --profile $Profile --region $Region --instance-ids $instanceId --output json | Out-Null
        if ($LASTEXITCODE -eq 0) {
            & $AwsPath ec2 wait instance-terminated --profile $Profile --region $Region --instance-ids $instanceId
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Error "EC2 termination could not be confirmed for $instanceId"
        }
        else {
            Write-Host "G2 instance terminated: $instanceId"
        }
    }
}
