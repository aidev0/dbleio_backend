#!/usr/bin/env python3
"""
Integrations API - Handle third-party integrations including Shopify
"""

from fastapi import APIRouter, HTTPException, status, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
import os
import httpx
import hmac
import hashlib
from urllib.parse import urlencode
from dotenv import load_dotenv
from pymongo import MongoClient
from src.auth import require_user_id, get_workos_user_id

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Shopify configuration
SHOPIFY_CLIENT_ID = os.getenv('SHOPIFY_CLIENT_ID')
SHOPIFY_CLIENT_SECRET = os.getenv('SHOPIFY_CLIENT_SECRET')
SHOPIFY_APP_URL = os.getenv('SHOPIFY_APP_URL', 'https://hpd-video-marketing-sim-c47311d71420.herokuapp.com')
SHOPIFY_SCOPES = os.getenv('SHOPIFY_SCOPES', 'read_products')

# Create routers
router = APIRouter(prefix="/api/integrations", tags=["integrations"])
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


# Pydantic Models
class Integration(BaseModel):
    """Integration model"""
    id: str = Field(alias="_id")
    user_id: str
    type: str  # 'shopify', 'meta', etc.
    store_url: Optional[str] = None
    shop_name: Optional[str] = None
    access_token: Optional[str] = None
    scopes: Optional[List[str]] = None
    status: str = "pending"  # pending, connected, disconnected, error
    properties: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class IntegrationCreate(BaseModel):
    """Create integration request"""
    user_id: str
    type: str
    store_url: Optional[str] = None


class ShopifyConnectRequest(BaseModel):
    """Request to initiate Shopify connection"""
    store_url: str  # e.g., "mystore.myshopify.com"
    # Note: user_id is now extracted from JWT, not from request body


# Helper functions
def verify_integration_access(integration_id: str, workos_user_id: str):
    """
    Verify user owns the integration.
    Returns the integration if access is granted, raises HTTPException otherwise.
    """
    integration = db.integrations.find_one({"_id": ObjectId(integration_id)})
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    if integration.get("user_id") != workos_user_id:
        raise HTTPException(status_code=403, detail="Access denied to this integration")

    return integration


def integration_helper(integration) -> dict:
    """Convert MongoDB integration to dict"""
    return {
        "_id": str(integration["_id"]),
        "user_id": integration["user_id"],
        "type": integration["type"],
        "store_url": integration.get("store_url"),
        "shop_name": integration.get("shop_name"),
        "access_token": integration.get("access_token"),
        "scopes": integration.get("scopes", []),
        "status": integration.get("status", "pending"),
        "properties": integration.get("properties", {}),
        "created_at": integration.get("created_at", datetime.utcnow()),
        "updated_at": integration.get("updated_at", datetime.utcnow())
    }


def normalize_shop_url(store_url: str) -> str:
    """Normalize shop URL to myshopify.com format"""
    store_url = store_url.strip().lower()
    # Remove protocol if present
    store_url = store_url.replace("https://", "").replace("http://", "")
    # Remove trailing slash
    store_url = store_url.rstrip("/")
    # Add .myshopify.com if not present
    if not store_url.endswith(".myshopify.com"):
        # Remove any existing domain suffix
        if "." in store_url:
            store_url = store_url.split(".")[0]
        store_url = f"{store_url}.myshopify.com"
    return store_url


# Routes

@router.get("/")
async def get_integrations(request: Request):
    """Get all integrations for the authenticated user"""
    try:
        # Get user ID from JWT
        workos_user_id = require_user_id(request)

        integrations = list(db.integrations.find({"user_id": workos_user_id}))
        return [integration_helper(i) for i in integrations]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_id}")
async def get_integration(integration_id: str, request: Request):
    """Get a specific integration (must be owner)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        integration = verify_integration_access(integration_id, workos_user_id)

        return integration_helper(integration)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{integration_id}")
async def delete_integration(integration_id: str, request: Request):
    """Delete an integration (must be owner)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        verify_integration_access(integration_id, workos_user_id)

        db.integrations.delete_one({"_id": ObjectId(integration_id)})
        return {"success": True, "message": "Integration deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Shopify OAuth Routes

@router.post("/shopify/connect")
async def initiate_shopify_connection(connect_request: ShopifyConnectRequest, request: Request):
    """
    Initiate Shopify OAuth flow
    Returns the authorization URL to redirect the user to
    """
    try:
        # Get user ID from JWT
        workos_user_id = require_user_id(request)

        shop = normalize_shop_url(connect_request.store_url)

        # Create or update integration record
        existing = db.integrations.find_one({
            "user_id": workos_user_id,
            "type": "shopify",
            "store_url": shop
        })

        if existing:
            integration_id = str(existing["_id"])
            db.integrations.update_one(
                {"_id": existing["_id"]},
                {"$set": {"status": "pending", "updated_at": datetime.utcnow()}}
            )
        else:
            new_integration = {
                "user_id": workos_user_id,
                "type": "shopify",
                "store_url": shop,
                "status": "pending",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = db.integrations.insert_one(new_integration)
            integration_id = str(result.inserted_id)

        # Build Shopify OAuth URL
        # State contains integration_id and user_id for callback
        state = f"{integration_id}:{workos_user_id}"

        redirect_uri = f"{SHOPIFY_APP_URL}/api/auth/shopify/callback"

        auth_params = {
            "client_id": SHOPIFY_CLIENT_ID,
            "scope": SHOPIFY_SCOPES,
            "redirect_uri": redirect_uri,
            "state": state,
        }

        auth_url = f"https://{shop}/admin/oauth/authorize?{urlencode(auth_params)}"

        return {
            "success": True,
            "auth_url": auth_url,
            "integration_id": integration_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/shopify/callback")
async def shopify_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    shop: str = Query(...),
    hmac_param: str = Query(None, alias="hmac"),
    timestamp: str = Query(None)
):
    """
    Handle Shopify OAuth callback
    Exchange authorization code for access token
    """
    try:
        # Parse state to get integration_id and user_id
        parts = state.split(":")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        integration_id, user_id = parts

        # Exchange code for access token
        token_url = f"https://{shop}/admin/oauth/access_token"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                json={
                    "client_id": SHOPIFY_CLIENT_ID,
                    "client_secret": SHOPIFY_CLIENT_SECRET,
                    "code": code
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to get access token: {response.text}"
                )

            token_data = response.json()
            access_token = token_data.get("access_token")
            scope = token_data.get("scope", "")

            # Get shop info
            shop_response = await client.get(
                f"https://{shop}/admin/api/2024-01/shop.json",
                headers={"X-Shopify-Access-Token": access_token}
            )

            shop_info = {}
            if shop_response.status_code == 200:
                shop_info = shop_response.json().get("shop", {})

        # Update integration with access token
        db.integrations.update_one(
            {"_id": ObjectId(integration_id)},
            {
                "$set": {
                    "access_token": access_token,
                    "scopes": scope.split(",") if scope else [],
                    "status": "connected",
                    "shop_name": shop_info.get("name", shop),
                    "properties": {
                        "shop_id": shop_info.get("id"),
                        "shop_email": shop_info.get("email"),
                        "shop_domain": shop_info.get("domain"),
                        "myshopify_domain": shop_info.get("myshopify_domain"),
                        "plan_name": shop_info.get("plan_name"),
                        "country": shop_info.get("country_name"),
                        "currency": shop_info.get("currency"),
                        "timezone": shop_info.get("timezone"),
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Redirect to frontend integrations page
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return RedirectResponse(url=f"{frontend_url}/app?tab=integrations&shopify=connected")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Shopify OAuth error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Shopify Data Fetching Routes

@router.get("/shopify/{integration_id}/products")
async def get_shopify_products(
    integration_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=250),
    page_info: Optional[str] = None
):
    """Fetch products from Shopify store (must own integration)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        integration = verify_integration_access(integration_id, workos_user_id)

        if integration.get("status") != "connected":
            raise HTTPException(status_code=400, detail="Integration not connected")

        access_token = integration.get("access_token")
        shop = integration.get("store_url")

        url = f"https://{shop}/admin/api/2024-01/products.json?limit={limit}"
        if page_info:
            url = f"https://{shop}/admin/api/2024-01/products.json?limit={limit}&page_info={page_info}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"X-Shopify-Access-Token": access_token}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch products: {response.text}"
                )

            data = response.json()

            # Parse pagination from Link header
            link_header = response.headers.get("Link", "")
            next_page_info = None
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        # Extract page_info from URL
                        import re
                        match = re.search(r'page_info=([^>&]+)', part)
                        if match:
                            next_page_info = match.group(1)

            return {
                "products": data.get("products", []),
                "next_page_info": next_page_info
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shopify/{integration_id}/orders")
async def get_shopify_orders(
    integration_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=250),
    status: str = Query("any"),  # any, open, closed, cancelled
    page_info: Optional[str] = None
):
    """Fetch orders from Shopify store (must own integration)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        integration = verify_integration_access(integration_id, workos_user_id)

        if integration.get("status") != "connected":
            raise HTTPException(status_code=400, detail="Integration not connected")

        access_token = integration.get("access_token")
        shop = integration.get("store_url")

        url = f"https://{shop}/admin/api/2024-01/orders.json?limit={limit}&status={status}"
        if page_info:
            url = f"https://{shop}/admin/api/2024-01/orders.json?limit={limit}&page_info={page_info}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"X-Shopify-Access-Token": access_token}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch orders: {response.text}"
                )

            data = response.json()

            # Parse pagination
            link_header = response.headers.get("Link", "")
            next_page_info = None
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        import re
                        match = re.search(r'page_info=([^>&]+)', part)
                        if match:
                            next_page_info = match.group(1)

            return {
                "orders": data.get("orders", []),
                "next_page_info": next_page_info
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shopify/{integration_id}/customers")
async def get_shopify_customers(
    integration_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=250),
    page_info: Optional[str] = None
):
    """Fetch customers from Shopify store (must own integration)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        integration = verify_integration_access(integration_id, workos_user_id)

        if integration.get("status") != "connected":
            raise HTTPException(status_code=400, detail="Integration not connected")

        access_token = integration.get("access_token")
        shop = integration.get("store_url")

        url = f"https://{shop}/admin/api/2024-01/customers.json?limit={limit}"
        if page_info:
            url = f"https://{shop}/admin/api/2024-01/customers.json?limit={limit}&page_info={page_info}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"X-Shopify-Access-Token": access_token}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch customers: {response.text}"
                )

            data = response.json()

            # Parse pagination
            link_header = response.headers.get("Link", "")
            next_page_info = None
            if 'rel="next"' in link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        import re
                        match = re.search(r'page_info=([^>&]+)', part)
                        if match:
                            next_page_info = match.group(1)

            return {
                "customers": data.get("customers", []),
                "next_page_info": next_page_info
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shopify/{integration_id}/analytics")
async def get_shopify_analytics(integration_id: str, request: Request):
    """Get basic analytics/counts from Shopify store (must own integration)"""
    try:
        # Verify user owns this integration
        workos_user_id = require_user_id(request)
        integration = verify_integration_access(integration_id, workos_user_id)

        if integration.get("status") != "connected":
            raise HTTPException(status_code=400, detail="Integration not connected")

        access_token = integration.get("access_token")
        shop = integration.get("store_url")

        async with httpx.AsyncClient() as client:
            # Get counts
            products_resp = await client.get(
                f"https://{shop}/admin/api/2024-01/products/count.json",
                headers={"X-Shopify-Access-Token": access_token}
            )
            orders_resp = await client.get(
                f"https://{shop}/admin/api/2024-01/orders/count.json?status=any",
                headers={"X-Shopify-Access-Token": access_token}
            )
            customers_resp = await client.get(
                f"https://{shop}/admin/api/2024-01/customers/count.json",
                headers={"X-Shopify-Access-Token": access_token}
            )

            return {
                "products_count": products_resp.json().get("count", 0) if products_resp.status_code == 200 else 0,
                "orders_count": orders_resp.json().get("count", 0) if orders_resp.status_code == 200 else 0,
                "customers_count": customers_resp.json().get("count", 0) if customers_resp.status_code == 200 else 0,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Auth router for Shopify OAuth callback at /api/auth/shopify/callback
@auth_router.get("/shopify/callback")
async def shopify_auth_callback(
    code: str = Query(...),
    state: str = Query(...),
    shop: str = Query(...),
    hmac_param: str = Query(None, alias="hmac"),
    timestamp: str = Query(None)
):
    """
    Handle Shopify OAuth callback at /api/auth/shopify/callback
    Exchange authorization code for access token
    """
    try:
        # Parse state to get integration_id and user_id
        parts = state.split(":")
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        integration_id, user_id = parts

        # Exchange code for access token
        token_url = f"https://{shop}/admin/oauth/access_token"

        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                token_url,
                json={
                    "client_id": SHOPIFY_CLIENT_ID,
                    "client_secret": SHOPIFY_CLIENT_SECRET,
                    "code": code
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to get access token: {response.text}"
                )

            token_data = response.json()
            access_token = token_data.get("access_token")
            scope = token_data.get("scope", "")

            # Get shop info
            shop_response = await http_client.get(
                f"https://{shop}/admin/api/2024-01/shop.json",
                headers={"X-Shopify-Access-Token": access_token}
            )

            shop_info = {}
            if shop_response.status_code == 200:
                shop_info = shop_response.json().get("shop", {})

        # Update integration with access token
        db.integrations.update_one(
            {"_id": ObjectId(integration_id)},
            {
                "$set": {
                    "access_token": access_token,
                    "scopes": scope.split(",") if scope else [],
                    "status": "connected",
                    "shop_name": shop_info.get("name", shop),
                    "properties": {
                        "shop_id": shop_info.get("id"),
                        "shop_email": shop_info.get("email"),
                        "shop_domain": shop_info.get("domain"),
                        "myshopify_domain": shop_info.get("myshopify_domain"),
                        "plan_name": shop_info.get("plan_name"),
                        "country": shop_info.get("country_name"),
                        "currency": shop_info.get("currency"),
                        "timezone": shop_info.get("timezone"),
                    },
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Redirect to frontend integrations page
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        return RedirectResponse(url=f"{frontend_url}/app?tab=integrations&shopify=connected")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Shopify OAuth error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
