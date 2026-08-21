[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AwsPath,

    [Parameter(Mandatory = $true)]
    [string]$Profile,

    [Parameter(Mandatory = $true)]
    [string]$Region,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedAccountId,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedImageDigest,

    [Parameter(Mandatory = $true)]
    [string]$LaunchTemplateId,

    [Parameter(Mandatory = $true)]
    [string]$BuildsBucket,

    [Parameter(Mandatory = $true)]
    [string]$OutputPrefix,

    [Parameter(Mandatory = $true)]
    [string]$EvidencePath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

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

$quotePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'cost-guard.json'
$quote = Get-Content -LiteralPath $quotePath -Raw | ConvertFrom-Json -Depth 20
if ($quote.region -ne $Region) {
    throw "Cost quote region mismatch: $($quote.region) != $Region"
}

$identity = Invoke-AwsJson sts get-caller-identity --region $Region
if ($identity.Account -ne $ExpectedAccountId) {
    throw "AWS account mismatch: $($identity.Account) != $ExpectedAccountId"
}

$plan = Invoke-AwsJson freetier get-account-plan-state --region us-east-1
if ($plan.accountPlanType -ne 'FREE' -or $plan.accountPlanStatus -ne 'ACTIVE') {
    throw "AWS Free plan is not active: type=$($plan.accountPlanType), status=$($plan.accountPlanStatus)"
}
if ($plan.accountPlanRemainingCredits.unit -ne 'USD') {
    throw "Unsupported credit unit: $($plan.accountPlanRemainingCredits.unit)"
}

$remainingCredits = [decimal]$plan.accountPlanRemainingCredits.amount
$initialCreditReference = [decimal]$quote.policy.credit_reference_usd
$reserve = $initialCreditReference * [decimal]$quote.policy.minimum_credit_fraction
$maximumRunCost = [decimal]$quote.calculation.maximum_gross_run_cost_usd
if ($remainingCredits -lt ($reserve + $maximumRunCost)) {
    throw "Credit guard failed: remaining=$remainingCredits USD, required reserve=$reserve USD plus run ceiling=$maximumRunCost USD"
}
if ([datetimeoffset]$plan.accountPlanExpirationDate -le [datetimeoffset]::UtcNow) {
    throw "AWS Free plan has expired"
}

$budget = Invoke-AwsJson budgets describe-budget --region us-east-1 --account-id $ExpectedAccountId --budget-name fireviewer-map-builder-monthly
$budgetLimit = [decimal]$budget.Budget.BudgetLimit.Amount
$budgetActual = if ($null -eq $budget.Budget.CalculatedSpend.ActualSpend) { [decimal]0 } else { [decimal]$budget.Budget.CalculatedSpend.ActualSpend.Amount }
$budgetStop = $budgetLimit * [decimal]$quote.policy.stop_at_budget_fraction
if ($budgetActual -ge $budgetStop) {
    throw "Monthly budget guard failed: actual=$budgetActual USD, stop threshold=$budgetStop USD"
}

$instanceType = Invoke-AwsJson ec2 describe-instance-types --region $Region --instance-types m7i-flex.large
$instance = $instanceType.InstanceTypes[0]
if (-not $instance.FreeTierEligible -or $instance.VCpuInfo.DefaultVCpus -ne 2 -or $instance.MemoryInfo.SizeInMiB -ne 8192) {
    throw "m7i-flex.large no longer matches the authorized Free Tier 2 vCPU / 8 GiB profile"
}
if ($instance.ProcessorInfo.SupportedArchitectures -notcontains 'x86_64') {
    throw "m7i-flex.large does not expose x86_64"
}

$quota = Invoke-AwsJson service-quotas get-service-quota --region $Region --service-code ec2 --quota-code L-1216C47A
if ([decimal]$quota.Quota.Value -lt 2) {
    throw "EC2 On-Demand vCPU quota is below two: $($quota.Quota.Value)"
}

$launchTemplate = Invoke-AwsJson ec2 describe-launch-template-versions --region $Region --launch-template-id $LaunchTemplateId --versions '$Latest'
$launchData = $launchTemplate.LaunchTemplateVersions[0].LaunchTemplateData
if ($launchData.InstanceType -ne 'm7i-flex.large') {
    throw "Launch template instance type drifted: $($launchData.InstanceType)"
}
$scratch = @($launchData.BlockDeviceMappings | Where-Object { $_.DeviceName -eq '/dev/sdf' })
if ($scratch.Count -ne 1 -or $scratch[0].Ebs.VolumeSize -lt 80 -or $scratch[0].Ebs.VolumeSize -gt 100 -or $scratch[0].Ebs.VolumeType -ne 'gp3' -or $scratch[0].Ebs.Iops -ne 3000 -or $scratch[0].Ebs.Throughput -ne 125 -or -not $scratch[0].Ebs.DeleteOnTermination) {
    throw "Launch template scratch volume drifted from the authorized gp3 profile"
}

$image = Invoke-AwsJson ecr describe-images --region $Region --repository-name fireviewer-map-builder --image-ids imageTag=reference-v1
$observedDigest = $image.imageDetails[0].imageDigest
if ($observedDigest -ne $ExpectedImageDigest) {
    throw "ECR digest mismatch: $observedDigest != $ExpectedImageDigest"
}

$activeInstances = Invoke-AwsJson ec2 describe-instances --region $Region --filters 'Name=tag:Application,Values=FireViewer' 'Name=tag:Component,Values=MapBuilder' 'Name=instance-state-name,Values=pending,running,stopping,stopped'
$activeInstanceIds = @(
    $activeInstances.Reservations |
        ForEach-Object { $_.Instances } |
        ForEach-Object { $_.InstanceId } |
        Where-Object { $_ }
)
if ($activeInstanceIds.Count -gt 0) {
    throw "A Map Builder instance already exists: $($activeInstanceIds -join ', ')"
}

$doneExists = $true
& $AwsPath s3api head-object --profile $Profile --region $Region --bucket $BuildsBucket --key "$OutputPrefix/zone.done.json" --output json 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    $doneExists = $false
}
if ($doneExists) {
    throw "The target build is already complete and immutable: s3://$BuildsBucket/$OutputPrefix/zone.done.json"
}

$ec2Rate = [decimal]$quote.rates.'m7i-flex.large_hour'
$gp3Rate = [decimal]$quote.rates.gp3_gib_month
$ipv4Rate = [decimal]$quote.rates.public_ipv4_hour
$hours = [decimal]$quote.calculation.maximum_runtime_hours
$hoursPerMonth = [decimal]$quote.calculation.hours_per_month
$volumeGiB = [decimal]$quote.calculation.root_volume_gib + [decimal]$quote.calculation.scratch_volume_gib
$calculatedGrossCost = ($ec2Rate * $hours) + (($gp3Rate * $volumeGiB / $hoursPerMonth) * $hours) + ($ipv4Rate * $hours)
if ($calculatedGrossCost -gt $maximumRunCost) {
    throw "Calculated G2 gross cost $calculatedGrossCost USD exceeds the authorized ceiling $maximumRunCost USD"
}

$evidence = [ordered]@{
    schema = 'fireviewer.aws-g2-preflight.v1'
    status = 'PASS'
    observed_at = [datetimeoffset]::UtcNow.ToString('o')
    account_id = $identity.Account
    account_plan = [ordered]@{
        type = $plan.accountPlanType
        status = $plan.accountPlanStatus
        remaining_credits_usd = $remainingCredits
        expiration = $plan.accountPlanExpirationDate
        protected_reserve_usd = $reserve
    }
    budget = [ordered]@{
        limit_usd = $budgetLimit
        actual_usd = $budgetActual
        stop_threshold_usd = $budgetStop
    }
    compute = [ordered]@{
        instance_type = $instance.InstanceType
        free_tier_eligible = $instance.FreeTierEligible
        vcpu = $instance.VCpuInfo.DefaultVCpus
        memory_mib = $instance.MemoryInfo.SizeInMiB
        architecture = 'x86_64'
        vcpu_quota = $quota.Quota.Value
        launch_template_id = $LaunchTemplateId
    }
    image_digest = $observedDigest
    target = "s3://$BuildsBucket/$OutputPrefix"
    maximum_runtime_hours = $hours
    calculated_known_gross_cost_usd = [math]::Round($calculatedGrossCost, 6)
    authorized_gross_cost_ceiling_usd = $maximumRunCost
}

$evidenceDirectory = Split-Path -Parent $EvidencePath
New-Item -ItemType Directory -Force -Path $evidenceDirectory | Out-Null
$evidence | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $EvidencePath -Encoding utf8NoBOM
$evidence | ConvertTo-Json -Depth 20
