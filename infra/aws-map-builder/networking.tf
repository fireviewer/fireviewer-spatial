data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "map_builder" {
  cidr_block           = "10.73.0.0/24"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = local.name_slug
  }
}

resource "aws_internet_gateway" "map_builder" {
  vpc_id = aws_vpc.map_builder.id

  tags = {
    Name = local.name_slug
  }
}

resource "aws_subnet" "worker" {
  vpc_id                  = aws_vpc.map_builder.id
  availability_zone       = data.aws_availability_zones.available.names[0]
  cidr_block              = "10.73.0.0/26"
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.name_slug}-worker"
  }
}

resource "aws_route_table" "worker" {
  vpc_id = aws_vpc.map_builder.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.map_builder.id
  }

  tags = {
    Name = "${local.name_slug}-worker"
  }
}

resource "aws_route_table_association" "worker" {
  subnet_id      = aws_subnet.worker.id
  route_table_id = aws_route_table.worker.id
}

# S3 traffic stays on the AWS backbone without a paid NAT gateway or interface
# endpoint. ECR, SSM and source HTTPS use the short-lived public IPv4 address.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.map_builder.id
  service_name      = "com.amazonaws.${local.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.worker.id]

  tags = {
    Name = "${local.name_slug}-s3"
  }
}

resource "aws_security_group" "worker" {
  name        = "${var.name_prefix}-worker"
  description = "No inbound access; DNS and outbound HTTPS only"
  vpc_id      = aws_vpc.map_builder.id

  egress {
    description = "HTTPS to AWS APIs, ECR and source services"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "DNS UDP to the VPC resolver"
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["${cidrhost(aws_vpc.map_builder.cidr_block, 2)}/32"]
  }

  egress {
    description = "DNS TCP to the VPC resolver"
    from_port   = 53
    to_port     = 53
    protocol    = "tcp"
    cidr_blocks = ["${cidrhost(aws_vpc.map_builder.cidr_block, 2)}/32"]
  }

  tags = {
    Name = "${local.name_slug}-worker"
  }
}
