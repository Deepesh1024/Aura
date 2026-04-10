# -----------------------------------------------------------------------------
# Aura — IAM Roles for EKS
# Provisions: EKS cluster role, node group role, and IRSA (IAM Roles for
# Service Accounts) with OIDC provider for pod-level AWS access.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# EKS Cluster IAM Role
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "eks_cluster_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_cluster" {
  name               = "${var.cluster_name}-cluster-role"
  assume_role_policy = data.aws_iam_policy_document.eks_cluster_assume.json

  tags = {
    Name = "${var.cluster_name}-cluster-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

resource "aws_iam_role_policy_attachment" "eks_service_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster.name
}

# -----------------------------------------------------------------------------
# EKS Node Group IAM Role
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "eks_node_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_node_group" {
  name               = "${var.cluster_name}-node-role"
  assume_role_policy = data.aws_iam_policy_document.eks_node_assume.json

  tags = {
    Name = "${var.cluster_name}-node-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group.name
}

resource "aws_iam_role_policy_attachment" "eks_ecr_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group.name
}

# -----------------------------------------------------------------------------
# OIDC Provider for IRSA (IAM Roles for Service Accounts)
# Allows Kubernetes pods to assume IAM roles via service account annotations.
# -----------------------------------------------------------------------------

data "tls_certificate" "eks" {
  url = aws_eks_cluster.aura.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.aura.identity[0].oidc[0].issuer

  tags = {
    Name = "${var.cluster_name}-oidc-provider"
  }
}

# -----------------------------------------------------------------------------
# IRSA Role — Aura Application
# Grants S3 and CloudWatch access to pods running with the 'aura-app' service account.
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "aura_app_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.eks.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(aws_eks_cluster.aura.identity[0].oidc[0].issuer, "https://", "")}:sub"
      values   = ["system:serviceaccount:aura:aura-app"]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(aws_eks_cluster.aura.identity[0].oidc[0].issuer, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "aura_app" {
  name               = "${var.cluster_name}-aura-app-role"
  assume_role_policy = data.aws_iam_policy_document.aura_app_assume.json

  tags = {
    Name = "${var.cluster_name}-aura-app-role"
  }
}

data "aws_iam_policy_document" "aura_app_permissions" {
  # S3 access for data storage
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject",
    ]
    resources = [
      aws_s3_bucket.aura_data.arn,
      "${aws_s3_bucket.aura_data.arn}/*",
    ]
  }

  # CloudWatch Logs access for observability
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "${aws_cloudwatch_log_group.eks.arn}:*",
    ]
  }
}

resource "aws_iam_policy" "aura_app" {
  name   = "${var.cluster_name}-aura-app-policy"
  policy = data.aws_iam_policy_document.aura_app_permissions.json

  tags = {
    Name = "${var.cluster_name}-aura-app-policy"
  }
}

resource "aws_iam_role_policy_attachment" "aura_app" {
  policy_arn = aws_iam_policy.aura_app.arn
  role       = aws_iam_role.aura_app.name
}
