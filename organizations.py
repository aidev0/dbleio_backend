#!/usr/bin/env python3
"""
Organizations API - Multi-tenancy support for the DBLE platform.
Organizations are embedded in the users collection as user.organizations[].
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from bson import ObjectId
import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Ensure indexes
db.organizations.create_index("slug", unique=True)

router = APIRouter(prefix="/api/organizations", tags=["organizations"])


# --- Models ---

class OrganizationCreate(BaseModel):
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    brand_name: Optional[str] = None
    brand_description: Optional[str] = None
    product_description: Optional[str] = None
    industry: Optional[str] = None
    logo_url: Optional[str] = None

class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    brand_name: Optional[str] = None
    brand_description: Optional[str] = None
    product_description: Optional[str] = None
    industry: Optional[str] = None
    logo_url: Optional[str] = None

class MemberAdd(BaseModel):
    user_id: str  # workos_user_id of user to add
    role: str = "member"  # owner, admin, member


# --- Helpers ---

ORG_FIELDS = [
    "name", "slug", "description", "url",
    "brand_name", "brand_description", "product_description",
    "industry", "logo_url",
    "created_by", "created_at", "updated_at",
]

def organization_helper(org) -> dict:
    return {
        "_id": str(org["_id"]),
        "name": org["name"],
        "slug": org["slug"],
        "description": org.get("description"),
        "url": org.get("url"),
        "brand_name": org.get("brand_name"),
        "brand_description": org.get("brand_description"),
        "product_description": org.get("product_description"),
        "industry": org.get("industry"),
        "logo_url": org.get("logo_url"),
        "created_by": org.get("created_by"),
        "created_at": org.get("created_at"),
        "updated_at": org.get("updated_at"),
    }


def _org_embed(org, role: str, joined_at: datetime) -> dict:
    """Build the embedded org dict stored in user.organizations[]."""
    data = organization_helper(org)
    data["role"] = role
    data["joined_at"] = joined_at
    return data


def _sync_org_to_users(org_id: str):
    """After an org is updated, refresh the embedded copy in every user that has it."""
    org = db.organizations.find_one({"_id": ObjectId(org_id)})
    if not org:
        return
    fresh = organization_helper(org)
    # Update all users who have this org embedded
    db.users.update_many(
        {"organizations._id": str(org["_id"])},
        {"$set": {
            "organizations.$[elem].name": fresh["name"],
            "organizations.$[elem].slug": fresh["slug"],
            "organizations.$[elem].description": fresh["description"],
            "organizations.$[elem].url": fresh["url"],
            "organizations.$[elem].brand_name": fresh["brand_name"],
            "organizations.$[elem].brand_description": fresh["brand_description"],
            "organizations.$[elem].product_description": fresh["product_description"],
            "organizations.$[elem].industry": fresh["industry"],
            "organizations.$[elem].logo_url": fresh["logo_url"],
            "organizations.$[elem].updated_at": fresh["updated_at"],
        }},
        array_filters=[{"elem._id": str(org["_id"])}],
    )


def _generate_slug(name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        slug = "org"
    base_slug = slug
    counter = 1
    while db.organizations.find_one({"slug": slug}):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_organization(body: OrganizationCreate, request: Request):
    """Create an organization and embed it in the creator's user doc."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        slug = body.slug if body.slug else _generate_slug(body.name)
        if db.organizations.find_one({"slug": slug}):
            raise HTTPException(status_code=409, detail=f"Slug '{slug}' already taken")

        now = datetime.utcnow()
        org_doc = {
            "name": body.name,
            "slug": slug,
            "description": body.description,
            "url": body.url,
            "brand_name": body.brand_name,
            "brand_description": body.brand_description,
            "product_description": body.product_description,
            "industry": body.industry,
            "logo_url": body.logo_url,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.organizations.insert_one(org_doc)
        org = db.organizations.find_one({"_id": result.inserted_id})

        # Embed into creator's user doc
        embed = _org_embed(org, role="owner", joined_at=now)
        db.users.update_one(
            {"workos_user_id": workos_user_id},
            {"$push": {"organizations": embed}},
        )

        return organization_helper(org)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_organizations(request: Request):
    """List organizations. Admins see all orgs, clients see only their own."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        user = db.users.find_one({"workos_user_id": workos_user_id})
        if not user:
            return []

        user_roles = user.get("roles", [])
        is_admin = "admin" in user_roles

        if is_admin:
            # Admins see all organizations
            all_orgs = list(db.organizations.find().sort("name", 1))
            return [organization_helper(o) for o in all_orgs]

        # Everyone else sees only their embedded orgs
        return user.get("organizations", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{org_id}")
async def get_organization(org_id: str, request: Request):
    """Get a single organization (must be a member)."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, org_id, workos_user_id)

        org = db.organizations.find_one({"_id": ObjectId(org_id)})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        return organization_helper(org)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{org_id}")
async def update_organization(org_id: str, body: OrganizationUpdate, request: Request):
    """Update an organization (owner/admin only). Syncs to all users."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, org_id, workos_user_id, required_roles=["owner", "admin"])

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        result = db.organizations.update_one(
            {"_id": ObjectId(org_id)},
            {"$set": update_dict}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Sync updated fields to all users who have this org
        _sync_org_to_users(org_id)

        org = db.organizations.find_one({"_id": ObjectId(org_id)})
        return organization_helper(org)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{org_id}", status_code=204)
async def delete_organization(org_id: str, request: Request):
    """Delete an organization (owner only). Removes from all users."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, org_id, workos_user_id, required_roles=["owner"])

        db.organizations.delete_one({"_id": ObjectId(org_id)})
        # Pull from all users
        db.users.update_many(
            {"organizations._id": org_id},
            {"$pull": {"organizations": {"_id": org_id}}},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{org_id}/members", status_code=201)
async def add_member(org_id: str, body: MemberAdd, request: Request):
    """Add a member to an organization (owner/admin only). Embeds org in target user."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, org_id, workos_user_id, required_roles=["owner", "admin"])

        org = db.organizations.find_one({"_id": ObjectId(org_id)})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")

        target_user = db.users.find_one({"workos_user_id": body.user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check not already a member
        existing_orgs = target_user.get("organizations", [])
        if any(o["_id"] == org_id for o in existing_orgs):
            raise HTTPException(status_code=409, detail="User is already a member")

        now = datetime.utcnow()
        embed = _org_embed(org, role=body.role, joined_at=now)
        db.users.update_one(
            {"workos_user_id": body.user_id},
            {"$push": {"organizations": embed}},
        )

        return embed
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{org_id}/members/{user_id}", status_code=204)
async def remove_member(org_id: str, user_id: str, request: Request):
    """Remove a member from an organization (owner/admin only, cannot remove last owner)."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, org_id, workos_user_id, required_roles=["owner", "admin"])

        target_user = db.users.find_one({"workos_user_id": user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        user_orgs = target_user.get("organizations", [])
        membership = next((o for o in user_orgs if o["_id"] == org_id), None)
        if not membership:
            raise HTTPException(status_code=404, detail="Membership not found")

        # Prevent removing the last owner
        if membership.get("role") == "owner":
            owner_count = db.users.count_documents({
                "organizations": {"$elemMatch": {"_id": org_id, "role": "owner"}}
            })
            if owner_count <= 1:
                raise HTTPException(status_code=400, detail="Cannot remove the last owner")

        db.users.update_one(
            {"workos_user_id": user_id},
            {"$pull": {"organizations": {"_id": org_id}}},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
