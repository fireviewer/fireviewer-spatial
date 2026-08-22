# FireViewer Map Builder on AWS

This directory provisions only the durable and control-plane foundation for
the restored Map Builder. It deliberately creates **no EC2 instance** and no
AWS Batch compute environment during the initial apply.

## Locked reference

- Builder Git commit: `766f157d00e15da72271ec197706c203f040fb7a`
- Git tag: `map-builder-reference-v1`
- Contract: `fireviewer.map-job.v1`
- Region: `eu-west-3`
- Compute: `m7i-flex.large` only
- Scratch: encrypted 100 GiB gp3, 3,000 IOPS, 125 MiB/s, delete on termination
- Container: Linux/amd64, pulled and executed by immutable digest

## Provisioned foundation

- two private, encrypted S3 buckets (transient work and versioned builds);
- immutable ECR repository;
- isolated VPC/public worker subnet, no inbound rules and no NAT gateway;
- free S3 gateway endpoint;
- least-privilege EC2 worker role plus SSM access;
- seven-day EC2/Batch log groups;
- monthly budget and optional percentage email notifications;
- a launch template with encrypted, delete-on-termination volumes;
- a GitHub OIDC role restricted to this repository and ECR publication.
- an optional Vercel Team OIDC role restricted to the production
  `fireviewer-api` project, immutable map requests, final S3 receipts and the
  single Map Builder Batch queue/definition.

The worker needs a short-lived public IPv4 address because it downloads public
geospatial sources and reaches ECR/SSM. There is no Elastic IP, bastion, load
balancer, NAT gateway or inbound listener.

## Apply

```powershell
terraform init `
  -backend-config='bucket=fireviewer-map-builds-640538430954-eu-west-3' `
  -backend-config='key=terraform/state/aws-map-builder.tfstate' `
  -backend-config='region=eu-west-3' `
  -backend-config='profile=unicorn-whodev' `
  -backend-config='encrypt=true' `
  -backend-config='use_lockfile=true'
terraform plan -var='aws_profile=unicorn-whodev' -out plans/foundation.tfplan
terraform apply plans/foundation.tfplan
```

The bounded G2 runner launches one instance from the launch template and
guarantees a termination attempt in its `finally` path.

Run the direct gate only after the immutable image is present in ECR:

```powershell
.\scripts\run-g2.ps1
```

The runner stops before launch unless the Free plan is active, enough credits
remain, the monthly budget is below 90%, the exact image digest is present,
the launch template still matches the 2 vCPU/8 GiB/gp3 profile, and no other
Map Builder instance is active.

G2/G3 passed on 2026-08-21. The versioned receipt is
`reference/map-builder-reference-v1/aws-g2-validation.json`.

## GitHub Actions and OIDC

On a `map-builder-*` Git tag or an explicit manual dispatch from `main`, the
`aws-map-builder-image` workflow runs runtime tests, builds Linux/amd64,
executes a container smoke test, pushes a unique immutable tag and records the
exact ECR digest. It uses temporary OIDC credentials and has no access to EC2,
Batch, S3, IAM administration or Hugging Face.

After applying the OIDC role, expose its non-secret ARN to GitHub as the
repository variable `AWS_MAP_BUILDER_ROLE_ARN`:

```powershell
terraform output -raw github_actions_role_arn
```

If `token.actions.githubusercontent.com` already exists in the AWS account,
set `github_oidc_provider_arn` to that provider ARN so Terraform does not try
to create a duplicate.

## AWS Batch activation

Batch is implemented but disabled by default. Enabling it requires both gates
to be explicit:

```powershell
terraform plan `
  -var='aws_profile=unicorn-whodev' `
  -var='g2_validated=true' `
  -var='enable_batch=true' `
  -var='batch_image_digest=sha256:...' `
  -var='vercel_team_slug=charli-dev420s-projects'
```

The Vercel team slug must come from the team URL, not from the internal
`team_...` identifier. With the Team issuer enabled in the Vercel project,
Terraform trusts only this exact subject:

```text
owner:<team-slug>:project:fireviewer-api:environment:production
```

The Batch worker is a generic Map Builder runtime. Incident-specific recovery
code such as `recovery_die_v2` remains in the backend and must never be copied,
called or configured in the worker image.

After apply, configure these non-secret production variables on
`fireviewer-api`, then redeploy the backend:

```text
AWS_ROLE_ARN                    = terraform output vercel_backend_role_arn
FV_MAP_PRODUCTION_PROVIDER     = aws_batch
FV_MAP_AWS_REGION              = eu-west-3
FV_MAP_AWS_WORK_BUCKET         = terraform output work_bucket_name
FV_MAP_AWS_BUILDS_BUCKET       = terraform output builds_bucket_name
FV_MAP_AWS_JOB_QUEUE           = terraform output batch_job_queue_arn
FV_MAP_AWS_JOB_DEFINITION      = terraform output batch_job_definition_arn
FV_MAP_AWS_TILE_SHARD_JOB_DEFINITION = terraform output batch_tile_shard_job_definition_arn
FV_MAP_AWS_MAX_PARALLEL_WORKERS = 8
FV_MAP_AWS_TARGET_TILES_PER_WORKER = 72
FV_MAP_AWS_BUILD_CREDIT_LIMIT_EUR = 2.0
FV_MAP_AWS_IMAGE_DIGEST        = sha256:ed75b3c253bae3441afc96f1f7524f69d2a5c34aa4a390a1c199380b392dde61
FV_MAP_AWS_BUILDER_GIT_COMMIT  = 488e75177e66c045840e358c2e3fd72cb14b560c
```

No AWS access key is stored in Vercel. The function exchanges its injected
OIDC token for short-lived role credentials.

The managed EC2 compute environment is locked to:

```text
min vCPU     = 0
desired vCPU = 0
max vCPU     = 16
instance     = m7i-flex.large
jobs         = 1 to 8 tile workers selected from the request tile count
```

It uses an ECS-optimised Amazon Linux 2023 worker, the same encrypted gp3
scratch profile and the same digest-pinned image. No job means no desired
worker capacity. The AWS-specific execution script remains separate from the
provider-neutral Python builder.

Requests up to 72 tiles keep the single-worker path. Heavier requests use
`ceil(tile_count / 72)` array children, capped at eight. Whole 4x4 source
metatiles stay on one child to avoid duplicate IGN downloads. A single
dependent assembler verifies and restores every checkpoint before producing
the final scene and tiled viewer. The Terraform cost guard budgets the maximum
worker time, assembler and exporter plus a 20% margin below EUR 2.00.

## Publication invariant

The container writes a provider-neutral output directory. The execution layer
performs one bulk artifact upload, compares the local/S3 object counts, then
writes and checksum-verifies `zone.done.json` last. It does not issue a slow
HEAD request for every artifact. A prefix without that final object is
incomplete.
