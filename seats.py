#!/usr/bin/env python3
"""
Seats API - Per-brand seat management.
Each seat links a user to a brand with a specific role.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
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

db.seats.create_index([("brand_id", 1), ("user_id", 1)], unique=True)

router = APIRouter(prefix="/api/seats", tags=["seats"])


# --- Models ---

class SeatCreate(BaseModel):
    brand_id: str
    user_id: str  # workos_user_id of the user to seat
    role: str = "editor"  # owner | admin | editor | viewer


class SeatUpdate(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None  # active | suspended


# --- Helpers ---

def seat_helper(doc) -> dict:
    return {
        "_id": str(doc["_id"]),
        "brand_id": doc["brand_id"],
        "organization_id": doc.get("organization_id"),
        "user_id": doc["user_id"],
        "role": doc.get("role", "editor"),
        "status": doc.get("status", "active"),
        "invited_by": doc.get("invited_by"),
        "invited_at": doc.get("invited_at"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def verify_brand_access(brand_id: str, workos_user_id: str, required_roles: list = None):
    """Verify user has a seat on the brand or is an org admin."""
    brand = db.brands.find_one({"_id": ObjectId(brand_id)})
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")

    # Check if platform admin
    user = db.users.find_one({"workos_user_id": workos_user_id})
    if user and "admin" in user.get("roles", []):
        return brand

    # Check org membership
    from role_helpers import verify_org_membership
    membership = verify_org_membership(db, brand["organization_id"], workos_user_id)

    # If org owner/admin, always has access
    if membership.get("role") in ["owner", "admin"]:
        return brand

    # Check seat-level access
    if required_roles:
        seat = db.seats.find_one({"brand_id": brand_id, "user_id": workos_user_id, "status": "active"})
        if not seat or seat.get("role") not in required_roles:
            raise HTTPException(status_code=403, detail=f"Requires brand role: {', '.join(required_roles)}")

    return brand


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_seat(body: SeatCreate, request: Request):
    """Add a seat to a brand (org owner/admin only)."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(body.brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        from role_helpers import verify_org_membership
        verify_org_membership(db, brand["organization_id"], workos_user_id, required_roles=["owner", "admin"])

        # Check target user exists
        target_user = db.users.find_one({"workos_user_id": body.user_id})
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check not already seated
        if db.seats.find_one({"brand_id": body.brand_id, "user_id": body.user_id}):
            raise HTTPException(status_code=409, detail="User already has a seat on this brand")

        now = datetime.utcnow()
        doc = {
            "brand_id": body.brand_id,
            "organization_id": brand["organization_id"],
            "user_id": body.user_id,
            "role": body.role,
            "status": "active",
            "invited_by": workos_user_id,
            "invited_at": now,
            "created_at": now,
            "updated_at": now,
        }
        result = db.seats.insert_one(doc)
        seat = db.seats.find_one({"_id": result.inserted_id})
        return seat_helper(seat)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_seats(request: Request, brand_id: Optional[str] = None, organization_id: Optional[str] = None):
    """List seats. Filter by brand_id or organization_id."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        query = {}
        if brand_id:
            verify_brand_access(brand_id, workos_user_id)
            query["brand_id"] = brand_id
        elif organization_id:
            from role_helpers import verify_org_membership
            verify_org_membership(db, organization_id, workos_user_id)
            query["organization_id"] = organization_id
        else:
            # Return seats for the current user
            query["user_id"] = workos_user_id

        seats = list(db.seats.find(query).sort("created_at", -1))
        return [seat_helper(s) for s in seats]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{seat_id}")
async def update_seat(seat_id: str, body: SeatUpdate, request: Request):
    """Update a seat role or status (org owner/admin only)."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        seat = db.seats.find_one({"_id": ObjectId(seat_id)})
        if not seat:
            raise HTTPException(status_code=404, detail="Seat not found")

        from role_helpers import verify_org_membership
        verify_org_membership(db, seat["organization_id"], workos_user_id, required_roles=["owner", "admin"])

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.seats.update_one({"_id": ObjectId(seat_id)}, {"$set": update_dict})

        updated = db.seats.find_one({"_id": ObjectId(seat_id)})
        return seat_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{seat_id}", status_code=204)
async def delete_seat(seat_id: str, request: Request):
    """Remove a seat (org owner/admin only)."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        seat = db.seats.find_one({"_id": ObjectId(seat_id)})
        if not seat:
            raise HTTPException(status_code=404, detail="Seat not found")

        from role_helpers import verify_org_membership
        verify_org_membership(db, seat["organization_id"], workos_user_id, required_roles=["owner", "admin"])

        db.seats.delete_one({"_id": ObjectId(seat_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
