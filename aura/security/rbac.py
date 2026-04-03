"""
RBAC — Role-Based Access Control for agent tools and data access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Permission:
    resource: str
    action: str

    def matches(self, required: str) -> bool:
        """Check if this permission grants access for the required permission string.
        Supports wildcard matching: 'read:*' matches 'read:retail_metrics'.
        """
        req_parts = required.split(":")
        perm_parts = f"{self.action}:{self.resource}".split(":")

        for req, perm in zip(req_parts, perm_parts):
            if perm == "*":
                continue
            if req != perm:
                return False
        return True



@dataclass
class Role:
    name: str
    description: str = ""
    permissions: dict[str, list[str]] = field(default_factory=dict)
    row_filter: str | None = None
    column_mask: list[str] = field(default_factory=list)

    def has_permission(self, category: str, required: str) -> bool:
        """Check if this role has the required permission."""
        perms = self.permissions.get(category, [])
        for perm in perms:
            # Wildcard match
            if perm.endswith(":*") and required.startswith(perm[:-1]):
                return True
            if perm == required:
                return True
        return False

    def mask_columns(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive columns based on role policy."""
        if not self.column_mask:
            return data
        return {
            k: "***MASKED***" if k in self.column_mask else v
            for k, v in data.items()
        }

    def apply_row_filter(self, sql: str) -> str:
        """Inject row-level filter into SQL query."""
        if not self.row_filter:
            return sql
        # Simple injection — production would use query AST rewriting
        if "WHERE" in sql.upper():
            return sql + f" AND ({self.row_filter})"
        return sql + f" WHERE {self.row_filter}"


class RBACManager:
    """
    Role-Based Access Control manager.

    Loads role definitions from YAML and provides permission-checking
    middleware for agent tools and data access.
    """

    def __init__(self, enabled: bool = True, default_role: str = "viewer") -> None:
        self.enabled = enabled
        self.default_role = default_role
        self._roles: dict[str, Role] = {}

    def load_policies(self, path: str | Path | None = None) -> None:
        """Load RBAC policies from a YAML file."""
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "rbac_policies.yaml"

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            for role_name, role_data in data.get("roles", {}).items():
                self._roles[role_name] = Role(
                    name=role_name,
                    description=role_data.get("description", ""),
                    permissions=role_data.get("permissions", {}),
                    row_filter=role_data.get("row_filter"),
                    column_mask=role_data.get("column_mask", []),
                )

            logger.info("Loaded %d RBAC roles from %s", len(self._roles), path)

        except FileNotFoundError:
            logger.warning("RBAC policy file not found: %s — using defaults", path)
            self._load_defaults()
        except Exception as e:
            logger.error("Failed to load RBAC policies: %s", e)
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default minimal RBAC roles."""
        self._roles = {
            "admin": Role(
                name="admin",
                description="Full access",
                permissions={
                    "data": ["read:*", "write:*"],
                    "agents": ["invoke:*"],
                    "actions": ["execute:*"],
                },
            ),
            "analyst": Role(
                name="analyst",
                description="Read + analysis",
                permissions={
                    "data": ["read:*"],
                    "agents": ["invoke:data_architect", "invoke:planner", "invoke:verifier"],
                    "actions": ["execute:safe_actions"],
                },
                column_mask=["ssn", "credit_card", "salary"],
            ),
            "viewer": Role(
                name="viewer",
                description="Read-only aggregated",
                permissions={
                    "data": ["read:aggregated"],
                    "agents": ["invoke:planner"],
                },
                row_filter="is_public = 1",
                column_mask=["ssn", "credit_card", "salary", "email", "phone", "address"],
            ),
        }

    def get_role(self, role_name: str) -> Role:
        """Get a role by name, falling back to default."""
        return self._roles.get(role_name, self._roles.get(self.default_role, Role(name="viewer")))

    def check_permission(
        self, role_name: str, category: str, action: str
    ) -> bool:
        """Check if a role has permission for an action."""
        if not self.enabled:
            return True
        role = self.get_role(role_name)
        return role.has_permission(category, action)

    def mask_data(
        self, role_name: str, data: dict[str, Any] | list[dict[str, Any]]
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Apply column masking based on role."""
        if not self.enabled:
            return data

        role = self.get_role(role_name)

        if isinstance(data, list):
            return [role.mask_columns(row) for row in data]
        return role.mask_columns(data)

    def filter_sql(self, role_name: str, sql: str) -> str:
        """Apply row-level filtering to SQL based on role."""
        if not self.enabled:
            return sql
        role = self.get_role(role_name)
        return role.apply_row_filter(sql)

    def list_roles(self) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name,
                "description": r.description,
                "masked_columns": r.column_mask,
                "has_row_filter": r.row_filter is not None,
            }
            for r in self._roles.values()
        ]
