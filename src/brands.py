#!/usr/bin/env python3
"""
Brands API - Each organization can have multiple brands.
A brand represents a single product/product line with its own URL and identity.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List
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

db.brands.create_index([("organization_id", 1), ("slug", 1)], unique=True)

router = APIRouter(prefix="/api/brands", tags=["brands"])


# --- Models ---

class BrandCreate(BaseModel):
    organization_id: str
    name: str
    slug: Optional[str] = None
    url: Optional[str] = None
    product_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    logo_url: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)


class BrandUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    product_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    logo_url: Optional[str] = None
    platforms: Optional[List[str]] = None


# --- Helpers ---

def brand_helper(doc) -> dict:
    return {
        "_id": str(doc["_id"]),
        "organization_id": doc["organization_id"],
        "name": doc["name"],
        "slug": doc.get("slug"),
        "url": doc.get("url"),
        "product_name": doc.get("product_name"),
        "description": doc.get("description"),
        "industry": doc.get("industry"),
        "logo_url": doc.get("logo_url"),
        "platforms": doc.get("platforms", []),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _generate_brand_slug(name: str, organization_id: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        slug = "brand"
    base_slug = slug
    counter = 1
    while db.brands.find_one({"organization_id": organization_id, "slug": slug}):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_brand(body: BrandCreate, request: Request):
    """Create a brand under an organization."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, body.organization_id, workos_user_id)

        slug = body.slug if body.slug else _generate_brand_slug(body.name, body.organization_id)
        if db.brands.find_one({"organization_id": body.organization_id, "slug": slug}):
            raise HTTPException(status_code=409, detail=f"Slug '{slug}' already taken in this organization")

        now = datetime.utcnow()
        doc = {
            "organization_id": body.organization_id,
            "name": body.name,
            "slug": slug,
            "url": body.url,
            "product_name": body.product_name,
            "description": body.description,
            "industry": body.industry,
            "logo_url": body.logo_url,
            "platforms": body.platforms,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.brands.insert_one(doc)
        brand = db.brands.find_one({"_id": result.inserted_id})
        return brand_helper(brand)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_brands(request: Request, organization_id: Optional[str] = None):
    """List brands. Filter by organization_id."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        query = {}
        if organization_id:
            from src.role_helpers import verify_org_membership
            verify_org_membership(db, organization_id, workos_user_id)
            query["organization_id"] = organization_id
        else:
            # Return brands for all orgs the user belongs to
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user:
                return []
            user_org_ids = [o["_id"] for o in user.get("organizations", [])]
            if "admin" in user.get("roles", []):
                pass  # admins see all
            else:
                query["organization_id"] = {"$in": user_org_ids}

        brands = list(db.brands.find(query).sort("name", 1))
        return [brand_helper(b) for b in brands]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}")
async def get_brand(brand_id: str, request: Request):
    """Get a single brand."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        verify_org_membership(db, brand["organization_id"], workos_user_id)
        return brand_helper(brand)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{brand_id}")
async def update_brand(brand_id: str, body: BrandUpdate, request: Request):
    """Update a brand (org owner/admin only)."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        verify_org_membership(db, brand["organization_id"], workos_user_id, required_roles=["owner", "admin"])

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.brands.update_one({"_id": ObjectId(brand_id)}, {"$set": update_dict})

        updated = db.brands.find_one({"_id": ObjectId(brand_id)})
        return brand_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{brand_id}", status_code=204)
async def delete_brand(brand_id: str, request: Request):
    """Delete a brand (org owner only)."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        verify_org_membership(db, brand["organization_id"], workos_user_id, required_roles=["owner"])

        db.brands.delete_one({"_id": ObjectId(brand_id)})
        # Also clean up seats for this brand
        db.seats.delete_many({"brand_id": brand_id})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
