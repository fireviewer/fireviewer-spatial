output "work_bucket_name" {
  value       = aws_s3_bucket.work.id
  description = "Transient requests/cache/failed bucket."
}

output "builds_bucket_name" {
  value       = aws_s3_bucket.builds.id
  description = "Versioned, durable map-build bucket."
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.map_builder.repository_url
  description = "Immutable ECR repository; execute images by digest."
}

output "direct_g2_launch_template_id" {
  value       = aws_launch_template.direct_g2.id
  description = "Launch template used by the bounded direct-EC2 runner."
}

output "direct_g2_launch_template_version" {
  value       = aws_launch_template.direct_g2.latest_version
  description = "Exact launch template revision to record in provenance."
}

output "worker_role_arn" {
  value       = aws_iam_role.worker.arn
  description = "Least-privilege temporary worker role."
}

output "batch_enabled" {
  value       = local.batch_activation_requested
  description = "Always false before G2/G3 are explicitly validated."
}

output "github_actions_role_arn" {
  value       = aws_iam_role.github_actions_ecr.arn
  description = "Set this non-secret ARN as the GitHub variable AWS_MAP_BUILDER_ROLE_ARN."
}

output "github_oidc_provider_arn" {
  value       = local.github_oidc_provider_arn
  description = "GitHub Actions OIDC provider used by the ECR publishing role."
}

output "batch_compute_environment_arn" {
  value       = try(aws_batch_compute_environment.map_builder[0].arn, null)
  description = "Managed EC2 Batch compute environment; null while Batch is disabled."
}

output "batch_job_queue_arn" {
  value       = try(aws_batch_job_queue.map_builder[0].arn, null)
  description = "Single-worker Batch queue; null while Batch is disabled."
}

output "batch_job_definition_arn" {
  value       = try(aws_batch_job_definition.map_builder[0].arn, null)
  description = "Digest-pinned Map Builder job definition; null while Batch is disabled."
}

output "batch_hf_exporter_job_definition_arn" {
  value       = try(aws_batch_job_definition.map_viewer_exporter[0].arn, null)
  description = "Pinned Batch job definition used only to publish validated tiled viewers to Hugging Face."
}

output "vercel_backend_role_arn" {
  value       = try(aws_iam_role.vercel_backend[0].arn, null)
  description = "Set this non-secret ARN as AWS_ROLE_ARN on the fireviewer-api Vercel project."
}

output "vercel_oidc_provider_arn" {
  value       = local.vercel_oidc_provider_arn
  description = "Team-scoped Vercel OIDC provider trusted by the backend role."
}

output "azure_backend_role_arn" {
  value       = try(aws_iam_role.azure_backend[0].arn, null)
  description = "Temporary AWS role assumed by the Azure Container Apps managed identity."
}

output "azure_oidc_provider_arn" {
  value       = try(aws_iam_openid_connect_provider.azure[0].arn, null)
  description = "Microsoft Entra OIDC provider trusted by the Azure backend role."
}
