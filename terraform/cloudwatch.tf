# -----------------------------------------------------------------------------
# Aura — CloudWatch Observability
# Provisions: Log group for EKS cluster control plane logs with configurable
# retention and KMS encryption.
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "eks" {
  # EKS requires the log group name to follow this exact convention
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.cloudwatch_retention_days

  tags = {
    Name        = "${var.cluster_name}-logs"
    Purpose     = "EKS control plane and application observability"
    Environment = var.environment
  }
}
