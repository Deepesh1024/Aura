# -----------------------------------------------------------------------------
# Aura — Terraform Variables
# Configurable parameters for the AWS infrastructure deployment.
# -----------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "aura-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.29"
}

variable "node_instance_type" {
  description = "EC2 instance type for EKS managed node group"
  type        = string
  default     = "m5.xlarge"
}

variable "node_desired_count" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "node_min_count" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "node_max_count" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 6
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for Aura data storage"
  type        = string
  default     = "aura-data-store"
}

variable "cloudwatch_retention_days" {
  description = "CloudWatch log group retention period in days"
  type        = number
  default     = 30
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default = {
    Project   = "Aura"
    ManagedBy = "Terraform"
  }
}
