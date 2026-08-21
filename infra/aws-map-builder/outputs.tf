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
