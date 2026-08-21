resource "aws_ecr_repository" "map_builder" {
  name                 = "fireviewer-map-builder"
  image_tag_mutability = "IMMUTABLE"

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "map_builder" {
  repository = aws_ecr_repository.map_builder.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Remove untagged layers after seven days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
