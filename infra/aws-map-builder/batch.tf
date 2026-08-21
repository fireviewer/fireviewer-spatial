locals {
  batch_activation_requested = var.enable_batch && var.g2_validated
}

resource "aws_launch_template" "batch" {
  count = local.batch_activation_requested ? 1 : 0

  name_prefix            = "${var.name_prefix}-batch-"
  update_default_version = true
  user_data              = filebase64("${path.module}/scripts/batch-bootstrap.mime")

  metadata_options {
    http_endpoint               = "enabled"
    http_protocol_ipv6          = "disabled"
    http_put_response_hop_limit = 2
    http_tokens                 = "required"
    instance_metadata_tags      = "enabled"
  }

  monitoring {
    enabled = false
  }

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      delete_on_termination = true
      encrypted             = true
      volume_size           = var.root_volume_size_gib
      volume_type           = "gp3"
    }
  }

  block_device_mappings {
    device_name = "/dev/sdf"

    ebs {
      delete_on_termination = true
      encrypted             = true
      iops                  = var.scratch_iops
      throughput            = var.scratch_throughput_mibps
      volume_size           = var.scratch_volume_size_gib
      volume_type           = "gp3"
    }
  }

  tag_specifications {
    resource_type = "instance"

    tags = merge(local.common_tags, {
      Name  = "${local.name_slug}-batch"
      Stage = "G4-batch"
    })
  }

  tag_specifications {
    resource_type = "volume"

    tags = merge(local.common_tags, {
      Name  = "${local.name_slug}-batch"
      Stage = "G4-batch"
    })
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_compute_environment" "map_builder" {
  count = local.batch_activation_requested ? 1 : 0

  name_prefix  = "${var.name_prefix}-"
  service_role = aws_iam_role.batch_service[0].arn
  state        = "ENABLED"
  type         = "MANAGED"

  compute_resources {
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    desired_vcpus       = 0
    instance_role       = aws_iam_instance_profile.batch_instance[0].arn
    instance_type       = [var.instance_type]
    max_vcpus           = 2
    min_vcpus           = 0
    security_group_ids  = [aws_security_group.worker.id]
    subnets             = [aws_subnet.worker.id]
    type                = "EC2"

    ec2_configuration {
      image_type = "ECS_AL2023"
    }

    launch_template {
      launch_template_id = aws_launch_template.batch[0].id
      version            = tostring(aws_launch_template.batch[0].latest_version)
    }

    tags = merge(local.common_tags, {
      Stage = "G4-batch"
    })
  }

  depends_on = [
    aws_iam_role_policy_attachment.batch_instance_ecs,
    aws_iam_role_policy_attachment.batch_service,
  ]

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_batch_job_queue" "map_builder" {
  count = local.batch_activation_requested ? 1 : 0

  name     = var.name_prefix
  priority = 1
  state    = "ENABLED"

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.map_builder[0].arn
  }
}

resource "aws_batch_job_definition" "map_builder" {
  count = local.batch_activation_requested ? 1 : 0

  name                  = var.name_prefix
  type                  = "container"
  platform_capabilities = ["EC2"]
  propagate_tags        = true

  parameters = {
    image_digest   = var.batch_image_digest
    output_s3_uri  = "REQUIRED"
    request_s3_uri = "REQUIRED"
  }

  container_properties = jsonencode({
    image            = "${aws_ecr_repository.map_builder.repository_url}@${var.batch_image_digest}"
    jobRoleArn       = aws_iam_role.batch_job[0].arn
    executionRoleArn = aws_iam_role.batch_execution[0].arn
    command = [
      "/opt/fireviewer/aws/execute-batch.sh",
      "--request-s3-uri",
      "Ref::request_s3_uri",
      "--output-s3-uri",
      "Ref::output_s3_uri",
      "--image-digest",
      "Ref::image_digest",
    ]
    environment = [
      { name = "AWS_DEFAULT_REGION", value = local.region },
      { name = "TMPDIR", value = "/scratch/tmp" },
      { name = "TMP", value = "/scratch/tmp" },
      { name = "TEMP", value = "/scratch/tmp" },
      { name = "XDG_CACHE_HOME", value = "/scratch/cache" },
    ]
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = local.region
        "awslogs-stream-prefix" = "job"
      }
    }
    mountPoints = [
      {
        sourceVolume  = "scratch"
        containerPath = "/scratch"
        readOnly      = false
      },
    ]
    readonlyRootFilesystem = false
    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "7168" },
    ]
    volumes = [
      {
        name = "scratch"
        host = { sourcePath = "/scratch/jobs" }
      },
    ]
  })

  retry_strategy {
    attempts = 2

    evaluate_on_exit {
      action       = "RETRY"
      on_exit_code = "75"
    }

    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "1"
    }

    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "2*"
    }

    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "3*"
    }

    evaluate_on_exit {
      action       = "EXIT"
      on_exit_code = "137"
    }

  }

  timeout {
    attempt_duration_seconds = 7200
  }
}
