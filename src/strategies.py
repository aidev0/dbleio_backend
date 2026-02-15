#!/usr/bin/env python3
"""
Strategies API - Strategies per campaign.
Each strategy defines budget and audience targeting for a campaign.
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

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# --- Models ---

class AudienceControl(BaseModel):
    location: List[str] = Field(default_factory=list)
    zip_codes: List[str] = Field(default_factory=list)
    in_market_interests: List[str] = Field(default_factory=list)


class PerformanceObjective(BaseModel):
    value: Optional[float] = None
    kpi: Optional[str] = None  # ROAS, CPA, CPL, CPC, CTR, etc.


class StrategyCreate(BaseModel):
    campaign_id: str
    name: str
    budget_amount: Optional[float] = None
    budget_type: Optional[str] = None  # daily, weekly, monthly
    performance_objective: Optional[PerformanceObjective] = Field(default_factory=PerformanceObjective)
    audience_control: Optional[AudienceControl] = Field(default_factory=AudienceControl)


class StrategyUpdate(BaseModel):
    name: Optional[str] = None
    budget_amount: Optional[float] = None
    budget_type: Optional[str] = None
    performance_objective: Optional[PerformanceObjective] = None
    audience_control: Optional[AudienceControl] = None


# --- Helpers ---

def strategy_helper(doc) -> dict:
    return {
        "_id": str(doc["_id"]),
        "campaign_id": doc["campaign_id"],
        "name": doc["name"],
        "budget_amount": doc.get("budget_amount"),
        "budget_type": doc.get("budget_type"),
        "performance_objective": doc.get("performance_objective", {}),
        "audience_control": doc.get("audience_control", {}),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _verify_campaign_access(campaign_id: str, workos_user_id: str):
    """Verify user has access to the campaign."""
    campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_owner = campaign.get("workos_user_id")
    shared_ids = campaign.get("shared_ids", [])
    if campaign_owner != workos_user_id and workos_user_id not in shared_ids:
        raise HTTPException(status_code=403, detail="Access denied to this campaign")
    return campaign


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_strategy(body: StrategyCreate, request: Request):
    """Create a strategy for a campaign."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_campaign_access(body.campaign_id, workos_user_id)

        now = datetime.utcnow()
        doc = {
            "campaign_id": body.campaign_id,
            "name": body.name,
            "budget_amount": body.budget_amount,
            "budget_type": body.budget_type,
            "performance_objective": body.performance_objective.model_dump() if body.performance_objective else {},
            "audience_control": body.audience_control.model_dump() if body.audience_control else {},
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.strategies.insert_one(doc)
        strategy = db.strategies.find_one({"_id": result.inserted_id})
        return strategy_helper(strategy)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_strategies(request: Request, campaign_id: Optional[str] = None):
    """List strategies. Filter by campaign_id."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        if campaign_id:
            _verify_campaign_access(campaign_id, workos_user_id)
            query = {"campaign_id": campaign_id}
        else:
            # Get all campaigns user has access to, then all strategies
            owned = [str(c["_id"]) for c in db.campaigns.find({"workos_user_id": workos_user_id}, {"_id": 1})]
            shared = [str(c["_id"]) for c in db.campaigns.find({"shared_ids": workos_user_id}, {"_id": 1})]
            campaign_ids = list(set(owned + shared))
            query = {"campaign_id": {"$in": campaign_ids}}

        strategies = list(db.strategies.find(query).sort("name", 1))
        return [strategy_helper(s) for s in strategies]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str, request: Request):
    """Get a single strategy."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        strategy = db.strategies.find_one({"_id": ObjectId(strategy_id)})
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        _verify_campaign_access(strategy["campaign_id"], workos_user_id)
        return strategy_helper(strategy)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, body: StrategyUpdate, request: Request):
    """Update a strategy."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        strategy = db.strategies.find_one({"_id": ObjectId(strategy_id)})
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        _verify_campaign_access(strategy["campaign_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        if "audience_control" in update_dict and update_dict["audience_control"] is not None:
            update_dict["audience_control"] = update_dict["audience_control"]

        update_dict["updated_at"] = datetime.utcnow()
        db.strategies.update_one({"_id": ObjectId(strategy_id)}, {"$set": update_dict})

        updated = db.strategies.find_one({"_id": ObjectId(strategy_id)})
        return strategy_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{strategy_id}", status_code=204)
async def delete_strategy(strategy_id: str, request: Request):
    """Delete a strategy."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        strategy = db.strategies.find_one({"_id": ObjectId(strategy_id)})
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        _verify_campaign_access(strategy["campaign_id"], workos_user_id)
        db.strategies.delete_one({"_id": ObjectId(strategy_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
