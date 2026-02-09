#!/usr/bin/env python3
"""
Role-based access control helpers for the DBLE platform.
"""

from fastapi import HTTPException, Request
from auth import require_user_id
from bson import ObjectId


def require_role(request: Request, db, allowed_roles: list) -> dict:
    """
    Verify the authenticated user has one of the allowed roles.
    Returns the user document or raises 403.
    """
    workos_user_id = require_user_id(request)
    user = db.users.find_one({"workos_user_id": workos_user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_roles = user.get("roles", [])
    if not any(role in user_roles for role in allowed_roles):
        raise HTTPException(
            status_code=403,
            detail=f"Requires one of roles: {', '.join(allowed_roles)}"
        )
    return user


def verify_org_membership(db, org_id: str, workos_user_id: str, required_roles: list = None) -> dict:
    """
    Check that a user has the organization embedded in their user doc.
    Platform admins bypass the membership check.
    Optionally check for specific org-level roles.
    Returns the embedded org dict or raises 403.
    """
    user = db.users.find_one({"workos_user_id": workos_user_id})
    if not user:
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    # Platform admins bypass org membership check
    if "admin" in user.get("roles", []):
        return {"_id": org_id, "role": "admin"}

    user_orgs = user.get("organizations", [])
    membership = next((o for o in user_orgs if o.get("_id") == org_id), None)
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    if required_roles:
        member_role = membership.get("role", "member")
        if member_role not in required_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires org role: {', '.join(required_roles)}"
            )
    return membership
