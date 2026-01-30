"""
Plans API
Handles pricing plans and subscriptions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
from pymongo import MongoClient

router = APIRouter(prefix="/api/plans", tags=["plans"])

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Default plans
DEFAULT_PLANS = [
    {
        "id": "platform",
        "name": "Platform",
        "price": 3000,
        "currency": "USD",
        "billing_period": "monthly",
        "commitment_months": 1,
        "description": "Platform access only. No builds included.",
        "features": {
            "platform_access": "full",
            "active_build_requests": 0,
            "parallel_builds": False,
            "onboarding": True,
            "cloud_llm_credits": "$1000 cloud + LLM usage credits included"
        },
        "best_for": [
            "Platform access only",
            "Self-service usage",
            "No dedicated builds"
        ],
        "compliance": ["SOC 2", "GDPR", "CCPA"],
        "security": ["Enterprise-grade Encryption", "Database", "Cloud", "PCI-Compliant Payments"],
        "active": True
    },
    {
        "id": "scale",
        "name": "SCALE",
        "price": 6000,
        "currency": "USD",
        "billing_period": "monthly",
        "commitment_months": 1,
        "description": "$250K+ annual ad spend. Lean team of 1-3 people.",
        "features": {
            "platform_access": "full",
            "active_build_requests": 1,
            "parallel_builds": False,
            "onboarding": True,
            "cloud_llm_credits": "$1000 cloud + LLM usage credits included",
            "dedicated_fem": 1
        },
        "best_for": [
            "$250K+ annual ad spend",
            "Lean team (1-3 people in marketing)",
            "Want to prove ROI before going all-in",
            "Need to solve one major bottleneck"
        ],
        "compliance": ["SOC 2", "GDPR", "CCPA"],
        "security": ["Enterprise-grade Encryption", "Database", "Cloud", "PCI-Compliant Payments"],
        "active": True
    },
    {
        "id": "enterprise",
        "name": "Enterprise",
        "price": None,
        "price_display": "Custom",
        "currency": "USD",
        "billing_period": "monthly",
        "commitment_months": None,
        "description": "Multiple brands. Custom requirements. Dedicated support.",
        "features": {
            "platform_access": "full",
            "active_build_requests": "3+",
            "parallel_builds": True,
            "onboarding": True,
            "cloud_llm_credits": "custom",
            "custom_solutions": True
        },
        "best_for": [
            "Operate multiple brands or sub-brands",
            "Sell across multiple marketplaces (Shopify + Amazon + TikTok Shop + own site)",
            "Need custom features not in standard platform",
            "Require faster iteration (multiple builds in parallel)",
            "Have compliance/security requirements"
        ],
        "compliance": ["SOC 2", "GDPR", "CCPA", "Custom Enterprise Compliances"],
        "security": ["Enterprise-grade Encryption", "Database", "Cloud", "PCI-Compliant Payments", "2FA", "Custom Enterprise Security"],
        "add_ons_available": ["fdm", "fde"],
        "active": True
    }
]

# Add-ons
ADD_ONS = [
    {
        "id": "fdm",
        "name": "Dedicated Account Manager (FDM)",
        "price": 4000,
        "currency": "USD",
        "billing_period": "monthly",
        "available_for": ["team", "enterprise"]
    },
    {
        "id": "fde",
        "name": "Additional Build Request (FDE)",
        "price": 4000,
        "currency": "USD",
        "billing_period": "monthly",
        "available_for": ["team", "enterprise"]
    }
]


@router.get("")
async def get_plans():
    """
    Get all active plans
    """
    try:
        plans = list(db.plans.find({"active": True}))
        for plan in plans:
            plan["_id"] = str(plan["_id"])
        return plans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/add-ons")
async def get_add_ons():
    """
    Get all available add-ons
    """
    try:
        add_ons = list(db.add_ons.find())
        for addon in add_ons:
            addon["_id"] = str(addon["_id"])
        return add_ons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{plan_id}")
async def get_plan(plan_id: str):
    """
    Get a specific plan by ID
    """
    try:
        plan = db.plans.find_one({"id": plan_id})
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        plan["_id"] = str(plan["_id"])
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SubscriptionCreate(BaseModel):
    user_id: str
    plan_id: str
    add_ons: Optional[List[str]] = []


@router.post("/subscribe")
async def create_subscription(subscription: SubscriptionCreate):
    """
    Create a new subscription for a user
    """
    try:
        # Verify plan exists
        plan = db.plans.find_one({"id": subscription.plan_id})
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")

        # Verify add-ons are valid for this plan
        if subscription.add_ons:
            plan_add_ons = plan.get("add_ons_available", [])
            for addon_id in subscription.add_ons:
                if addon_id not in plan_add_ons:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Add-on '{addon_id}' is not available for plan '{subscription.plan_id}'"
                    )

        # Calculate total price
        total_price = plan["price"]
        for addon_id in subscription.add_ons:
            addon = db.add_ons.find_one({"id": addon_id})
            if addon:
                total_price += addon["price"]

        # Create subscription document
        doc = {
            "user_id": subscription.user_id,
            "plan_id": subscription.plan_id,
            "plan_name": plan["name"],
            "add_ons": subscription.add_ons,
            "total_price": total_price,
            "currency": plan["currency"],
            "billing_period": plan["billing_period"],
            "commitment_months": plan["commitment_months"],
            "status": "pending",  # Will be updated after payment
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = db.subscriptions.insert_one(doc)

        return {
            "subscription_id": str(result.inserted_id),
            "plan": plan["name"],
            "total_price": total_price,
            "currency": plan["currency"],
            "message": "Subscription created. Pending payment."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/subscription")
async def get_user_subscription(user_id: str):
    """
    Get the active subscription for a user
    """
    try:
        subscription = db.subscriptions.find_one(
            {"user_id": user_id, "status": "active"},
            sort=[("created_at", -1)]
        )
        if not subscription:
            return None
        subscription["_id"] = str(subscription["_id"])
        return subscription
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
