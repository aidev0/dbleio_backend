#!/usr/bin/env python3
"""
Research API — endpoints for the Research stage of the content generator pipeline.
Provides brand analysis, Instagram analytics, competitive analysis, trends, and financial data.
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from src.auth import require_user_id, get_workos_user_id
from src.task_manager import task_manager, TaskStatus
from src.research_helpers import (
    get_top_performers,
    build_trend_data,
    extract_brand_from_url,
    fetch_financial_data,
    extract_assets_for_top_reels,
    analyze_video_content,
)

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/research", tags=["research"])


# --- Request models ---

class BrandUrlRequest(BaseModel):
    url: str


class AnalyzeVideoRequest(BaseModel):
    video_url: str


class RunResearchRequest(BaseModel):
    brand_url: Optional[str] = None
    brand_username: Optional[str] = None
    competitor_usernames: Optional[list] = None
    financial_companies: Optional[list] = None


# --- Helpers ---

def _get_workflow(workflow_id: str) -> dict:
    from bson import ObjectId
    wf = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


def _save_research_to_workflow(workflow_id: str, key: str, data: dict):
    """Save research data into workflow.config.stage_settings.research.<key>"""
    from bson import ObjectId
    db.content_workflows.update_one(
        {"_id": ObjectId(workflow_id)},
        {"$set": {
            f"config.stage_settings.research.{key}": data,
            "updated_at": datetime.utcnow(),
        }},
    )


# --- Endpoints ---

@router.post("/{workflow_id}/run")
async def run_research(
    workflow_id: str,
    body: RunResearchRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Kick off full research pipeline as a background task."""
    user_id = get_workos_user_id(request) or "api_key"
    _get_workflow(workflow_id)

    task_id = f"research_{workflow_id}_{uuid.uuid4().hex[:8]}"
    task_manager.create_task(task_id, "research", metadata={
        "workflow_id": workflow_id,
        "user_id": user_id,
    })

    background_tasks.add_task(
        _run_research_background,
        workflow_id,
        task_id,
        body.brand_url,
        body.brand_username,
        body.competitor_usernames or [],
        body.financial_companies or [],
    )

    return {"task_id": task_id, "status": "started"}


@router.get("/{workflow_id}/status")
async def get_research_status(workflow_id: str, task_id: Optional[str] = None, request: Request = None):
    """Poll research progress. If task_id is provided, check that specific task."""
    if task_id:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    # Find most recent research task for this workflow
    all_tasks = task_manager._tasks
    research_tasks = [
        t for t in all_tasks.values()
        if t.get("task_type") == "research"
        and t.get("metadata", {}).get("workflow_id") == workflow_id
    ]
    if not research_tasks:
        return {"status": "none", "message": "No research tasks found"}

    latest = max(research_tasks, key=lambda t: t.get("created_at", ""))
    return latest


@router.get("/{workflow_id}/results")
async def get_research_results(workflow_id: str, request: Request = None):
    """Get completed research data from the workflow."""
    wf = _get_workflow(workflow_id)
    config = wf.get("config", {})
    stage_settings = config.get("stage_settings", {})
    research = stage_settings.get("research", {})
    return research


@router.post("/{workflow_id}/analyze-brand-url")
async def analyze_brand_url(
    workflow_id: str,
    body: BrandUrlRequest,
    request: Request,
):
    """Fetch & analyze a brand website."""
    get_workos_user_id(request)  # auth check (API key or JWT)
    _get_workflow(workflow_id)

    result = extract_brand_from_url(body.url)
    if "error" not in result:
        _save_research_to_workflow(workflow_id, "brand_url_analysis", result)

    return result


@router.get("/{workflow_id}/brand-instagram")
async def get_brand_instagram(
    workflow_id: str,
    username: str,
    request: Request = None,
):
    """Brand's Instagram analytics + top 20%."""
    _get_workflow(workflow_id)
    result = get_top_performers(username, percentile=0.2)
    if "error" not in result:
        _save_research_to_workflow(workflow_id, "brand_instagram", result)
    return result


@router.get("/{workflow_id}/competitor-instagram")
async def get_competitor_instagram(
    workflow_id: str,
    username: str,
    request: Request = None,
):
    """Competitor's Instagram analytics + top 20%."""
    _get_workflow(workflow_id)
    result = get_top_performers(username, percentile=0.2)
    if "error" not in result:
        _save_research_to_workflow(
            workflow_id,
            f"competitor_instagram.{username}",
            result,
        )
    return result


@router.post("/{workflow_id}/analyze-video")
async def analyze_video(
    workflow_id: str,
    body: AnalyzeVideoRequest,
    request: Request,
):
    """Deep Gemini analysis of a single video."""
    get_workos_user_id(request)  # auth check (API key or JWT)
    _get_workflow(workflow_id)

    result = analyze_video_content(body.video_url, proxy=True)
    return result


@router.post("/{workflow_id}/extract-assets")
async def extract_assets(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    username: Optional[str] = None,
    max_reels: int = 10,
    frames_per_reel: int = 4,
):
    """Extract frames from top reels, upload to GCS. Background task."""
    user_id = require_user_id(request)
    _get_workflow(workflow_id)

    task_id = f"extract_assets_{workflow_id}_{uuid.uuid4().hex[:8]}"
    task_manager.create_task(task_id, "extract_assets", metadata={
        "workflow_id": workflow_id,
        "username": username,
    })

    background_tasks.add_task(
        _extract_assets_background, workflow_id, task_id,
        username or "stitchfix", max_reels, frames_per_reel,
    )
    return {"task_id": task_id, "status": "started"}


@router.get("/{workflow_id}/trends")
async def get_engagement_trends(
    workflow_id: str,
    username: str,
    request: Request = None,
):
    """Time-series engagement data (likes/comments/views by date)."""
    _get_workflow(workflow_id)
    result = build_trend_data(username)
    _save_research_to_workflow(workflow_id, f"trends.{username}", result)
    return result


@router.get("/{workflow_id}/financial")
async def get_financial_data_endpoint(
    workflow_id: str,
    company: Optional[str] = None,
    request: Request = None,
):
    """Financial data for specified companies."""
    _get_workflow(workflow_id)

    companies = []
    if company:
        companies = [company]
    else:
        # Default: StitchFix and RentTheRunway
        companies = ["Stitch Fix", "Rent The Runway"]

    results = {}
    for co in companies:
        key = co.lower().replace(" ", "_")
        data = fetch_financial_data(co)
        results[key] = data

    _save_research_to_workflow(workflow_id, "financial", results)
    return results


# --- Background tasks ---

def _extract_assets_background(
    workflow_id: str,
    task_id: str,
    username: str,
    max_reels: int,
    frames_per_reel: int,
):
    """Background task that extracts frames from top reels and uploads to GCS."""
    try:
        task_manager.update_task(
            task_id, status=TaskStatus.PROCESSING, progress=5,
            message=f"Fetching top reels for @{username}...",
        )

        # Get top performers
        top_data = get_top_performers(username, percentile=0.2)
        if "error" in top_data:
            task_manager.update_task(
                task_id, status=TaskStatus.FAILED,
                message=f"Failed: {top_data['error']}",
                error=top_data["error"],
            )
            return

        top_performers = top_data.get("top_performers", [])
        reels_with_video = [p for p in top_performers if p.get("videoUrl")][:max_reels]

        task_manager.update_task(
            task_id, progress=10,
            message=f"Extracting frames from {len(reels_with_video)} reels...",
        )

        for i, reel in enumerate(reels_with_video):
            short_code = reel.get("shortCode", "unknown")
            task_manager.update_task(
                task_id,
                progress=10 + int((i / max(len(reels_with_video), 1)) * 80),
                message=f"Extracting frames from reel {i+1}/{len(reels_with_video)} ({short_code})...",
            )

            from src.research_helpers import extract_reel_frames
            frames = extract_reel_frames(
                video_url=reel["videoUrl"],
                short_code=short_code,
                username=username,
                num_frames=frames_per_reel,
            )
            reel["extracted_frames"] = frames

        # Save updated top performers with frames back to workflow
        top_data["top_performers"] = top_performers
        _save_research_to_workflow(workflow_id, f"brand_instagram", top_data)

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message=f"Extracted frames from {len(reels_with_video)} reels",
            result={"workflow_id": workflow_id, "reels_processed": len(reels_with_video)},
        )

    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"Asset extraction failed: {str(e)}",
            error=str(e),
        )


def _run_research_background(
    workflow_id: str,
    task_id: str,
    brand_url: Optional[str],
    brand_username: Optional[str],
    competitor_usernames: list,
    financial_companies: list,
):
    """Background task that runs the full research pipeline."""
    from src.research_prompts import COMPETITIVE_SUCCESS_PROMPT

    try:
        task_manager.update_task(
            task_id, status=TaskStatus.PROCESSING, progress=5,
            message="Starting research...",
        )

        # Step 1: Brand URL analysis (if provided)
        if brand_url:
            task_manager.update_task(
                task_id, progress=10, message="Analyzing brand website...",
            )
            brand_analysis = extract_brand_from_url(brand_url)
            if "error" not in brand_analysis:
                _save_research_to_workflow(workflow_id, "brand_url_analysis", brand_analysis)

        # Step 2: Brand Instagram
        if brand_username:
            task_manager.update_task(
                task_id, progress=20, message=f"Analyzing @{brand_username} Instagram...",
            )
            brand_ig = get_top_performers(brand_username, percentile=0.2)
            if "error" not in brand_ig:
                _save_research_to_workflow(workflow_id, "brand_instagram", brand_ig)

                # AI analysis on top 5 reels
                top_reels = [
                    p for p in brand_ig.get("top_performers", [])
                    if p.get("type") == "reel" and p.get("videoUrl")
                ][:5]
                for i, reel in enumerate(top_reels):
                    task_manager.update_task(
                        task_id, progress=25 + i * 3,
                        message=f"AI analyzing brand reel {i+1}/{len(top_reels)}...",
                    )
                    ai = analyze_video_content(reel["videoUrl"], proxy=True)
                    if "error" not in ai:
                        reel["ai_analysis"] = ai

                _save_research_to_workflow(workflow_id, "brand_instagram", brand_ig)

            # Brand trends
            task_manager.update_task(
                task_id, progress=40, message="Building brand trend data...",
            )
            brand_trends = build_trend_data(brand_username)
            _save_research_to_workflow(workflow_id, f"trends.{brand_username}", brand_trends)

        # Step 3: Competitor Instagram
        for ci, comp_username in enumerate(competitor_usernames):
            task_manager.update_task(
                task_id, progress=45 + ci * 10,
                message=f"Analyzing competitor @{comp_username}...",
            )
            comp_ig = get_top_performers(comp_username, percentile=0.2)
            if "error" not in comp_ig:
                # AI analysis on top 5 competitor reels
                top_reels = [
                    p for p in comp_ig.get("top_performers", [])
                    if p.get("type") == "reel" and p.get("videoUrl")
                ][:5]
                for j, reel in enumerate(top_reels):
                    task_manager.update_task(
                        task_id,
                        message=f"AI analyzing @{comp_username} reel {j+1}/{len(top_reels)}...",
                    )
                    ai = analyze_video_content(reel["videoUrl"], proxy=True)
                    if "error" not in ai:
                        reel["ai_analysis"] = ai

                # Competitive success analysis
                if brand_username:
                    brand_ig_data = get_top_performers(brand_username, percentile=0.2)
                    brand_profile = db.instagram_profiles.find_one({"username": brand_username})
                    comp_profile = db.instagram_profiles.find_one({"username": comp_username})

                    brand_summary = "\n".join([
                        f"- {p.get('caption', '')[:100]} (likes: {p.get('likesCount', 0)}, comments: {p.get('commentsCount', 0)}, views: {p.get('videoPlayCount', 0)})"
                        for p in brand_ig_data.get("top_performers", [])[:10]
                    ])
                    comp_summary = "\n".join([
                        f"- {p.get('caption', '')[:100]} (likes: {p.get('likesCount', 0)}, comments: {p.get('commentsCount', 0)}, views: {p.get('videoPlayCount', 0)})"
                        for p in comp_ig.get("top_performers", [])[:10]
                    ])

                    from src.research_helpers import _call_llm
                    prompt = COMPETITIVE_SUCCESS_PROMPT.format(
                        brand_username=brand_username,
                        brand_followers=brand_profile.get("followersCount", 0) if brand_profile else 0,
                        competitor_username=comp_username,
                        competitor_followers=comp_profile.get("followersCount", 0) if comp_profile else 0,
                        brand_content_summary=brand_summary,
                        top_content_summary=comp_summary,
                    )
                    success_analysis = _call_llm(prompt)
                    if success_analysis:
                        comp_ig["success_analysis"] = success_analysis

                _save_research_to_workflow(
                    workflow_id,
                    f"competitor_instagram.{comp_username}",
                    comp_ig,
                )

                # Competitor trends
                comp_trends = build_trend_data(comp_username)
                _save_research_to_workflow(workflow_id, f"trends.{comp_username}", comp_trends)

        # Step 4: Financial data
        if financial_companies:
            task_manager.update_task(
                task_id, progress=80, message="Fetching financial data...",
            )
            financial_results = {}
            for co in financial_companies:
                key = co.lower().replace(" ", "_")
                financial_results[key] = fetch_financial_data(co)
            _save_research_to_workflow(workflow_id, "financial", financial_results)

        # Done
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Research complete",
            result={"workflow_id": workflow_id},
        )

    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"Research failed: {str(e)}",
            error=str(e),
        )
