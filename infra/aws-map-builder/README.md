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
- a launch template with encrypted, delete-on-termination volumes.

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

Do not set `enable_batch=true`. Terraform rejects that setting while
`g2_validated=false`. The bounded G2 runner launches one instance from the
launch template and guarantees a termination attempt in its `finally` path.

Run the direct gate only after the immutable image is present in ECR:

```powershell
.\scripts\run-g2.ps1
```

The runner stops before launch unless the Free plan is active, enough credits
remain, the monthly budget is below 90%, the exact image digest is present,
the launch template still matches the 2 vCPU/8 GiB/gp3 profile, and no other
Map Builder instance is active.

## Publication invariant

The container writes a provider-neutral output directory. The execution layer
uploads artifacts, verifies their S3 metadata, uploads manifests, and writes
`zone.done.json` last. A prefix without that final object is incomplete.
