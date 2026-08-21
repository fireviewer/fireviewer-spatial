data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_launch_template" "direct_g2" {
  name_prefix            = "${var.name_prefix}-g2-"
  image_id               = data.aws_ssm_parameter.al2023_ami.value
  instance_type          = var.instance_type
  update_default_version = true

  instance_initiated_shutdown_behavior = "terminate"
  user_data                            = filebase64("${path.module}/scripts/bootstrap.sh")

  iam_instance_profile {
    name = aws_iam_instance_profile.worker.name
  }

  network_interfaces {
    associate_public_ip_address = true
    delete_on_termination       = true
    device_index                = 0
    security_groups             = [aws_security_group.worker.id]
    subnet_id                   = aws_subnet.worker.id
  }

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
      Name  = "${local.name_slug}-g2"
      Stage = "G2-direct"
    })
  }

  tag_specifications {
    resource_type = "volume"

    tags = merge(local.common_tags, {
      Name  = "${local.name_slug}-g2"
      Stage = "G2-direct"
    })
  }

  lifecycle {
    create_before_destroy = true
  }
}
