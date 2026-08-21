resource "aws_cloudwatch_log_group" "ec2" {
  name              = "/aws/ec2/fireviewer-map-builder"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/fireviewer-map-builder"
  retention_in_days = 7
}
