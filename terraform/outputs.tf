# -----------------------------------------------------------------------------
# Aura — Terraform Outputs
# Useful values for downstream configuration and CI/CD integration.
# -----------------------------------------------------------------------------

output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.aura.name
}

output "eks_cluster_endpoint" {
  description = "API endpoint for the EKS cluster"
  value       = aws_eks_cluster.aura.endpoint
}

output "eks_cluster_version" {
  description = "Kubernetes version of the EKS cluster"
  value       = aws_eks_cluster.aura.version
}

output "eks_cluster_certificate_authority" {
  description = "Base64-encoded certificate data for cluster authentication"
  value       = aws_eks_cluster.aura.certificate_authority[0].data
  sensitive   = true
}

output "s3_bucket_arn" {
  description = "ARN of the Aura data storage S3 bucket"
  value       = aws_s3_bucket.aura_data.arn
}

output "s3_bucket_name" {
  description = "Name of the Aura data storage S3 bucket"
  value       = aws_s3_bucket.aura_data.id
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for EKS"
  value       = aws_cloudwatch_log_group.eks.name
}

output "oidc_provider_arn" {
  description = "ARN of the OIDC provider for IRSA"
  value       = aws_iam_openid_connect_provider.eks.arn
}

output "aura_app_role_arn" {
  description = "IAM role ARN for the Aura application service account"
  value       = aws_iam_role.aura_app.arn
}

output "kubeconfig_command" {
  description = "AWS CLI command to update kubeconfig for cluster access"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.aura.name}"
}
