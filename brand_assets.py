#!/usr/bin/env python3
"""
Brand Assets API - Files, images, videos, logos, and other media per brand.
Supports file upload to GCS.
"""

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from google.cloud import storage
from google.oauth2 import service_account
import json

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

GCS_BUCKET = os.getenv('GCS_BUCKET', 'dble-input-videos')

# Init GCS
def _get_gcs_credentials():
    creds_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if creds_json:
        try:
            creds_info = json.loads(creds_json)
            return service_account.Credentials.from_service_account_info(creds_info)
        except Exception as e:
            print(f"Error loading service account from env: {e}")
    return None

try:
    _gcs_credentials = _get_gcs_credentials()
    if _gcs_credentials:
        storage_client = storage.Client(credentials=_gcs_credentials, project=_gcs_credentials.project_id)
    else:
        storage_client = storage.Client()
    gcs_bucket = storage_client.bucket(GCS_BUCKET)
    print(f"✓ brand_assets GCS initialized")
except Exception as e:
    print(f"Warning: brand_assets GCS init failed: {e}")
    storage_client = None
    gcs_bucket = None

router = APIRouter(prefix="/api/brand-assets", tags=["brand-assets"])


# --- Models ---

class BrandAssetCreate(BaseModel):
    brand_id: str
    name: str
    asset_type: str = "file"
    url: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None


class BrandAssetUpdate(BaseModel):
    name: Optional[str] = None
    asset_type: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None


# --- Helpers ---

def asset_helper(doc) -> dict:
    # For GCS assets, generate fresh signed URL; never return gs:// or stale URLs
    url = None
    if doc.get("gcs_blob_name") and gcs_bucket:
        url = _get_signed_url(doc["gcs_blob_name"])
    elif doc.get("url") and not str(doc.get("url", "")).startswith("gs://"):
        url = doc.get("url")
    return {
        "_id": str(doc["_id"]),
        "brand_id": doc["brand_id"],
        "name": doc["name"],
        "asset_type": doc.get("asset_type", "file"),
        "url": url,
        "file_name": doc.get("file_name"),
        "file_size": doc.get("file_size"),
        "content_type": doc.get("content_type"),
        "gs_uri": doc.get("gs_uri"),
        "description": doc.get("description"),
        "metadata": doc.get("metadata"),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _verify_brand_membership(brand_id: str, workos_user_id: str, required_roles: list = None):
    brand = db.brands.find_one({"_id": ObjectId(brand_id)})
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    from role_helpers import verify_org_membership
    verify_org_membership(db, brand["organization_id"], workos_user_id, required_roles=required_roles)
    return brand


def _get_signed_url(blob_name: str, expiration_hours: int = 1) -> str:
    """Generate a fresh signed URL for a GCS blob (1 hour default)."""
    try:
        credentials = _get_gcs_credentials()
        blob = gcs_bucket.blob(blob_name)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="GET",
            credentials=credentials,
        )
    except Exception as e:
        print(f"Warning: signed URL generation failed: {e}")
        return None


def _detect_asset_type(content_type: str) -> str:
    if not content_type:
        return "file"
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    if content_type.startswith("audio/"):
        return "audio"
    if content_type == "application/pdf":
        return "document"
    if "font" in content_type:
        return "font"
    return "file"


# --- Endpoints ---

@router.post("/upload", status_code=201)
async def upload_brand_asset(
    request: Request,
    file: UploadFile = File(...),
    brand_id: str = Form(...),
    name: str = Form(""),
    description: str = Form(""),
    asset_type: str = Form(""),
):
    """Upload a file to GCS and create a brand asset record."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_brand_membership(brand_id, workos_user_id)

        if not gcs_bucket:
            raise HTTPException(status_code=500, detail="GCS not configured")

        # Use filename as name if not provided
        asset_name = name.strip() or file.filename or "Untitled"
        detected_type = asset_type.strip() or _detect_asset_type(file.content_type)

        # Upload to GCS
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = file.filename.replace(" ", "_") if file.filename else "file"
        blob_name = f"brand-assets/{brand_id}/{timestamp}_{safe_filename}"
        blob = gcs_bucket.blob(blob_name)
        blob.upload_from_file(file.file, content_type=file.content_type)

        now = datetime.utcnow()
        doc = {
            "brand_id": brand_id,
            "name": asset_name,
            "asset_type": detected_type,
            "url": None,
            "gs_uri": f"gs://{GCS_BUCKET}/{blob_name}",
            "gcs_blob_name": blob_name,
            "file_name": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "description": description.strip() or None,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.brand_assets.insert_one(doc)
        asset = db.brand_assets.find_one({"_id": result.inserted_id})
        return asset_helper(asset)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_brand_asset(body: BrandAssetCreate, request: Request):
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_brand_membership(body.brand_id, workos_user_id)

        now = datetime.utcnow()
        doc = {
            "brand_id": body.brand_id,
            "name": body.name,
            "asset_type": body.asset_type,
            "url": body.url,
            "description": body.description,
            "metadata": body.metadata,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.brand_assets.insert_one(doc)
        asset = db.brand_assets.find_one({"_id": result.inserted_id})
        return asset_helper(asset)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_brand_assets(request: Request, brand_id: Optional[str] = None):
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        if brand_id:
            _verify_brand_membership(brand_id, workos_user_id)
            query = {"brand_id": brand_id}
        else:
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user:
                return []
            user_org_ids = [o["_id"] for o in user.get("organizations", [])]
            if "admin" in user.get("roles", []):
                query = {}
            else:
                brand_ids = [str(b["_id"]) for b in db.brands.find({"organization_id": {"$in": user_org_ids}}, {"_id": 1})]
                query = {"brand_id": {"$in": brand_ids}}

        assets_list = list(db.brand_assets.find(query).sort("created_at", -1))
        return [asset_helper(a) for a in assets_list]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{asset_id}")
async def get_brand_asset(asset_id: str, request: Request):
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        asset = db.brand_assets.find_one({"_id": ObjectId(asset_id)})
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        _verify_brand_membership(asset["brand_id"], workos_user_id)
        return asset_helper(asset)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{asset_id}")
async def update_brand_asset(asset_id: str, body: BrandAssetUpdate, request: Request):
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        asset = db.brand_assets.find_one({"_id": ObjectId(asset_id)})
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        _verify_brand_membership(asset["brand_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.brand_assets.update_one({"_id": ObjectId(asset_id)}, {"$set": update_dict})

        updated = db.brand_assets.find_one({"_id": ObjectId(asset_id)})
        return asset_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{asset_id}", status_code=204)
async def delete_brand_asset(asset_id: str, request: Request):
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        asset = db.brand_assets.find_one({"_id": ObjectId(asset_id)})
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        _verify_brand_membership(asset["brand_id"], workos_user_id)

        # Delete from GCS if uploaded
        if asset.get("gcs_blob_name") and gcs_bucket:
            try:
                blob = gcs_bucket.blob(asset["gcs_blob_name"])
                blob.delete()
            except Exception:
                pass  # File may already be deleted

        db.brand_assets.delete_one({"_id": ObjectId(asset_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
