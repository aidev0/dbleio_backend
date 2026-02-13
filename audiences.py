#!/usr/bin/env python3
"""
Audiences API - Target audiences per brand.
Each audience defines demographics and targeting for campaigns.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/audiences", tags=["audiences"])


# --- Models ---

class Demographics(BaseModel):
    age_range: Optional[List[int]] = None  # [min, max]
    gender: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    income_level: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    behaviors: List[str] = Field(default_factory=list)


class AudienceCreate(BaseModel):
    brand_id: str
    name: str
    description: Optional[str] = None
    demographics: Demographics = Field(default_factory=Demographics)
    size_estimate: Optional[int] = None


class AudienceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    demographics: Optional[Demographics] = None
    size_estimate: Optional[int] = None


# --- Helpers ---

def audience_helper(doc) -> dict:
    return {
        "_id": str(doc["_id"]),
        "brand_id": doc["brand_id"],
        "name": doc["name"],
        "description": doc.get("description"),
        "demographics": doc.get("demographics", {}),
        "size_estimate": doc.get("size_estimate"),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _verify_brand_membership(brand_id: str, workos_user_id: str, required_roles: list = None):
    """Verify user has access to the brand's organization."""
    brand = db.brands.find_one({"_id": ObjectId(brand_id)})
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    from role_helpers import verify_org_membership
    verify_org_membership(db, brand["organization_id"], workos_user_id, required_roles=required_roles)
    return brand


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_audience(body: AudienceCreate, request: Request):
    """Create an audience for a brand."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_brand_membership(body.brand_id, workos_user_id)

        now = datetime.utcnow()
        doc = {
            "brand_id": body.brand_id,
            "name": body.name,
            "description": body.description,
            "demographics": body.demographics.model_dump() if body.demographics else {},
            "size_estimate": body.size_estimate,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.audiences.insert_one(doc)
        audience = db.audiences.find_one({"_id": result.inserted_id})
        return audience_helper(audience)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_audiences(request: Request, brand_id: Optional[str] = None):
    """List audiences. Filter by brand_id."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        if brand_id:
            _verify_brand_membership(brand_id, workos_user_id)
            query = {"brand_id": brand_id}
        else:
            # Get all brands user has access to, then all audiences
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user:
                return []
            user_org_ids = [o["_id"] for o in user.get("organizations", [])]
            if "admin" in user.get("roles", []):
                query = {}
            else:
                brand_ids = [str(b["_id"]) for b in db.brands.find({"organization_id": {"$in": user_org_ids}}, {"_id": 1})]
                query = {"brand_id": {"$in": brand_ids}}

        audiences = list(db.audiences.find(query).sort("name", 1))
        return [audience_helper(a) for a in audiences]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{audience_id}")
async def get_audience(audience_id: str, request: Request):
    """Get a single audience."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        audience = db.audiences.find_one({"_id": ObjectId(audience_id)})
        if not audience:
            raise HTTPException(status_code=404, detail="Audience not found")

        _verify_brand_membership(audience["brand_id"], workos_user_id)
        return audience_helper(audience)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{audience_id}")
async def update_audience(audience_id: str, body: AudienceUpdate, request: Request):
    """Update an audience."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        audience = db.audiences.find_one({"_id": ObjectId(audience_id)})
        if not audience:
            raise HTTPException(status_code=404, detail="Audience not found")

        _verify_brand_membership(audience["brand_id"], workos_user_id, required_roles=["owner", "admin"])

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        if "demographics" in update_dict and update_dict["demographics"] is not None:
            update_dict["demographics"] = update_dict["demographics"]

        update_dict["updated_at"] = datetime.utcnow()
        db.audiences.update_one({"_id": ObjectId(audience_id)}, {"$set": update_dict})

        updated = db.audiences.find_one({"_id": ObjectId(audience_id)})
        return audience_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{audience_id}", status_code=204)
async def delete_audience(audience_id: str, request: Request):
    """Delete an audience."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        audience = db.audiences.find_one({"_id": ObjectId(audience_id)})
        if not audience:
            raise HTTPException(status_code=404, detail="Audience not found")

        _verify_brand_membership(audience["brand_id"], workos_user_id, required_roles=["owner", "admin"])
        db.audiences.delete_one({"_id": ObjectId(audience_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
