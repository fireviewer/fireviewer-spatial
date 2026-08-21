# Intentionally empty until G2/G3 pass. Keeping this file and the variable gate
# makes the migration sequence explicit without creating a queue, compute
# environment or persistent capacity prematurely. The validated container and
# launch settings will be reused here after g2_validated is deliberately set.

locals {
  batch_activation_requested = var.enable_batch && var.g2_validated
}
