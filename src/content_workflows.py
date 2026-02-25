#!/usr/bin/env python3
"""
Content Workflows API routes.
Exposes the content generation orchestrator to the frontend.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta
from bson import ObjectId
import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/content/workflows", tags=["content-workflows"])


# --- Models ---

class ContentWorkflowCreate(BaseModel):
    brand_id: str
    title: str
    description: Optional[str] = None
    config: Optional[dict] = None


class ContentWorkflowUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[dict] = None


class ApprovalRequest(BaseModel):
    approved: bool
    note: Optional[str] = None


class HumanInputRequest(BaseModel):
    input_data: dict = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    message: str
    role: str = "user"


class GenerateConceptsRequest(BaseModel):
    num: int = Field(default=3, ge=1, le=50)
    tone: str = "engaging"
    content_type: str = "reel"


class GenerateStoryboardRequest(BaseModel):
    concept_index: int
    llm_model: Optional[str] = None  # "gemini-pro-3" | "claude-sonnet" | etc.
    image_model: Optional[str] = None  # default image model for assets


class GenerateStoryboardImageRequest(BaseModel):
    concept_index: int
    variation_index: int = 0
    target_type: str  # "character" | "scene"
    target_id: str
    image_model: Optional[str] = None


class GenerateConceptImageRequest(BaseModel):
    concept_index: int
    image_model: Optional[str] = None
    slide_index: Optional[int] = None  # for carousels
    content_piece_key: Optional[str] = None  # to scope storage


class GenerateVideoRequest(BaseModel):
    storyboard_index: int = 0  # which storyboard to use
    count: int = 1  # how many creatives
    model: str  # video model id
    output_format: Optional[str] = None  # reel_9_16, story_9_16, etc.
    resolution: Optional[str] = None  # 720p, 1080p, 4k
    temperature: Optional[float] = None
    custom_prompt: Optional[str] = None  # user-provided prompt template


# --- Helpers ---


def _build_visual_style_context(research: dict) -> str:
    """
    Extract visual style insights from research data for storyboard/image generation.
    Focuses on colors, textures, objects, characters, lighting, and shot types.
    """
    from collections import Counter

    colors_all, textures_all, objects_all, characters_all = [], [], [], []
    formats_all, music_all = [], []

    def _collect_from_performers(performers: list):
        for p in performers:
            ai = p.get("ai_analysis")
            if not ai or "error" in ai:
                continue
            if ai.get("colors"):
                colors_all.extend(ai["colors"] if isinstance(ai["colors"], list) else [ai["colors"]])
            if ai.get("textures"):
                textures_all.extend(ai["textures"] if isinstance(ai["textures"], list) else [ai["textures"]])
            if ai.get("objects"):
                objects_all.extend(ai["objects"] if isinstance(ai["objects"], list) else [ai["objects"]])
            if ai.get("content_format"):
                formats_all.append(ai["content_format"])
            if ai.get("music_style"):
                music_all.append(ai["music_style"])
            if ai.get("characters") and isinstance(ai["characters"], list):
                for c in ai["characters"]:
                    if isinstance(c, dict):
                        characters_all.append(f"{c.get('gender', '')} {c.get('age_range', '')} — {c.get('description', '')}")
                    elif isinstance(c, str):
                        characters_all.append(c)

    # Collect from brand and competitor performers
    brand_ig = research.get("brand_instagram")
    if brand_ig and isinstance(brand_ig, dict):
        _collect_from_performers(brand_ig.get("top_performers", []))
    comp_ig = research.get("competitor_instagram", {})
    if isinstance(comp_ig, dict):
        for comp_data in comp_ig.values():
            if isinstance(comp_data, dict):
                _collect_from_performers(comp_data.get("top_performers", []))

    lines = []
    color_counts = Counter(colors_all)
    if color_counts:
        lines.append(f"Dominant colors in top content: {', '.join(c for c, _ in color_counts.most_common(8))}")
    texture_counts = Counter(textures_all)
    if texture_counts:
        lines.append(f"Visual textures: {', '.join(t for t, _ in texture_counts.most_common(6))}")
    obj_counts = Counter(objects_all)
    if obj_counts:
        lines.append(f"Common objects/props: {', '.join(o for o, _ in obj_counts.most_common(8))}")
    if characters_all:
        lines.append(f"Character types seen: {'; '.join(characters_all[:5])}")
    fmt_counts = Counter(formats_all)
    if fmt_counts:
        lines.append(f"Top content formats: {', '.join(f for f, _ in fmt_counts.most_common(4))}")

    if not lines:
        return ""
    return "VISUAL STYLE FROM RESEARCH (top-performing content in this niche):\n" + "\n".join(lines)


def _build_research_context(research: dict) -> str:
    """
    Build a rich, structured context string from research data for concept generation.
    Extracts AI video analysis patterns, brand insights, competitor insights, and trends.
    """
    parts = []

    # --- Brand URL analysis ---
    brand_url = research.get("brand_url_analysis")
    if brand_url and isinstance(brand_url, dict) and "error" not in brand_url:
        brand_bits = []
        for key in ("brand_voice", "messaging_themes", "target_audience", "products", "colors", "fonts"):
            val = brand_url.get(key)
            if val:
                brand_bits.append(f"  {key}: {val if isinstance(val, str) else ', '.join(val) if isinstance(val, list) else str(val)}")
        if brand_bits:
            parts.append("BRAND IDENTITY (from website):\n" + "\n".join(brand_bits))

    # --- Extract AI analysis patterns from top performers ---
    def _extract_ai_patterns(performers: list, label: str) -> str:
        """Summarize AI analysis patterns across top performers."""
        analyzed = [p for p in performers if p.get("ai_analysis") and "error" not in p.get("ai_analysis", {})]
        if not analyzed:
            return ""

        hooks, formats, durations, colors_all, textures_all, objects_all = [], [], [], [], [], []
        music_styles, ctas, characters_all, transcripts = [], [], [], []

        for p in analyzed:
            ai = p["ai_analysis"]
            if ai.get("hook_type"):
                hooks.append(ai["hook_type"])
            if ai.get("hook_text"):
                hooks.append(f'"{ai["hook_text"]}"')
            if ai.get("content_format"):
                formats.append(ai["content_format"])
            if ai.get("duration"):
                durations.append(str(ai["duration"]))
            if ai.get("colors"):
                colors_all.extend(ai["colors"] if isinstance(ai["colors"], list) else [ai["colors"]])
            if ai.get("textures"):
                textures_all.extend(ai["textures"] if isinstance(ai["textures"], list) else [ai["textures"]])
            if ai.get("objects"):
                objects_all.extend(ai["objects"] if isinstance(ai["objects"], list) else [ai["objects"]])
            if ai.get("music_style"):
                music_styles.append(ai["music_style"])
            if ai.get("cta"):
                ctas.append(ai["cta"])
            if ai.get("characters") and isinstance(ai["characters"], list):
                for c in ai["characters"]:
                    if isinstance(c, dict):
                        characters_all.append(f"{c.get('gender', '')} {c.get('age_range', '')} — {c.get('description', '')}")
                    elif isinstance(c, str):
                        characters_all.append(c)
            if ai.get("transcription"):
                transcripts.append(ai["transcription"][:150])

        lines = [f"{label} — {len(analyzed)} reels analyzed:"]

        # Count frequencies for hooks and formats
        from collections import Counter
        hook_counts = Counter([h for h in hooks if not h.startswith('"')])
        if hook_counts:
            lines.append(f"  Hook types: {', '.join(f'{h} ({c}x)' for h, c in hook_counts.most_common(5))}")
        hook_examples = [h for h in hooks if h.startswith('"')][:3]
        if hook_examples:
            lines.append(f"  Hook examples: {'; '.join(hook_examples)}")

        fmt_counts = Counter(formats)
        if fmt_counts:
            lines.append(f"  Content formats: {', '.join(f'{f} ({c}x)' for f, c in fmt_counts.most_common(5))}")

        if durations:
            lines.append(f"  Durations: {', '.join(set(durations))}")

        color_counts = Counter(colors_all)
        if color_counts:
            lines.append(f"  Dominant colors: {', '.join(c for c, _ in color_counts.most_common(8))}")

        texture_counts = Counter(textures_all)
        if texture_counts:
            lines.append(f"  Textures: {', '.join(t for t, _ in texture_counts.most_common(6))}")

        obj_counts = Counter(objects_all)
        if obj_counts:
            lines.append(f"  Common objects: {', '.join(o for o, _ in obj_counts.most_common(8))}")

        music_counts = Counter(music_styles)
        if music_counts:
            lines.append(f"  Music styles: {', '.join(m for m, _ in music_counts.most_common(5))}")

        if characters_all:
            lines.append(f"  Characters seen: {'; '.join(characters_all[:6])}")

        cta_counts = Counter(c for c in ctas if c and c.lower() != "none")
        if cta_counts:
            lines.append(f"  CTAs used: {', '.join(c for c, _ in cta_counts.most_common(5))}")

        if transcripts:
            lines.append(f"  Sample transcripts: {' | '.join(transcripts[:2])}")

        return "\n".join(lines)

    # --- Brand Instagram ---
    brand_ig = research.get("brand_instagram")
    if brand_ig and isinstance(brand_ig, dict) and "error" not in brand_ig:
        username = brand_ig.get("username", "brand")
        followers = brand_ig.get("followers", 0)
        total = brand_ig.get("total_reels", 0)
        top = brand_ig.get("top_performers", [])
        parts.append(f"BRAND INSTAGRAM @{username}: {followers:,} followers, {total} reels, top {len(top)} performers")

        # Top performer stats
        if top:
            avg_views = sum(p.get("videoPlayCount", 0) or 0 for p in top) / len(top)
            avg_likes = sum(p.get("likesCount", 0) or 0 for p in top) / len(top)
            parts.append(f"  Top performer avg: {avg_views:,.0f} views, {avg_likes:,.0f} likes")

        ai_summary = _extract_ai_patterns(top, f"BRAND AI ANALYSIS (@{username})")
        if ai_summary:
            parts.append(ai_summary)

    # --- Competitor Instagram ---
    comp_ig = research.get("competitor_instagram", {})
    if isinstance(comp_ig, dict):
        for comp_user, comp_data in comp_ig.items():
            if not comp_data or not isinstance(comp_data, dict) or "error" in comp_data:
                continue
            followers = comp_data.get("followers", 0)
            total = comp_data.get("total_reels", 0)
            top = comp_data.get("top_performers", [])
            parts.append(f"COMPETITOR @{comp_user}: {followers:,} followers, {total} reels, top {len(top)} performers")

            if top:
                avg_views = sum(p.get("videoPlayCount", 0) or 0 for p in top) / len(top)
                avg_likes = sum(p.get("likesCount", 0) or 0 for p in top) / len(top)
                parts.append(f"  Top performer avg: {avg_views:,.0f} views, {avg_likes:,.0f} likes")

            ai_summary = _extract_ai_patterns(top, f"COMPETITOR AI ANALYSIS (@{comp_user})")
            if ai_summary:
                parts.append(ai_summary)

            # Success analysis
            success = comp_data.get("success_analysis")
            if success and isinstance(success, str):
                parts.append(f"COMPETITIVE SUCCESS INSIGHT (@{comp_user}):\n  {success[:400]}")

    # --- Financial context ---
    financial = research.get("financial", {})
    if isinstance(financial, dict):
        for co_key, fin_data in financial.items():
            if not fin_data or not isinstance(fin_data, dict):
                continue
            bits = []
            for k in ("revenue", "market_cap", "stock_price", "employees", "revenue_growth"):
                v = fin_data.get(k)
                if v:
                    bits.append(f"{k}: {v}")
            if bits:
                parts.append(f"FINANCIAL ({co_key}): {', '.join(bits)}")

    if not parts:
        return ""
    return "--- RESEARCH DATA ---\n" + "\n\n".join(parts) + "\n--- END RESEARCH DATA ---"


def _workflow_helper(doc) -> dict:
    if not doc:
        return {}
    return {
        "_id": str(doc["_id"]),
        "brand_id": doc.get("brand_id"),
        "organization_id": doc.get("organization_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "status": doc.get("status"),
        "current_stage": doc.get("current_stage"),
        "current_stage_index": doc.get("current_stage_index", 0),
        "config": doc.get("config", {}),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _refresh_media_urls(output_data: dict) -> dict:
    """Re-sign GCS URLs in output_data so media never expires for the client."""
    from src.videos import get_fresh_signed_url

    # Re-sign video variations
    for var in output_data.get("variations", []):
        gs_uri = var.get("gs_uri")
        if gs_uri:
            fresh = get_fresh_signed_url(gs_uri)
            if fresh:
                var["preview"] = fresh

    # Re-sign storyboard character & scene images
    for sb in output_data.get("storyboards", []):
        for char in sb.get("characters", []):
            gs_uri = char.get("gs_uri")
            if gs_uri:
                fresh = get_fresh_signed_url(gs_uri)
                if fresh:
                    char["image_url"] = fresh
        for scene in sb.get("scenes", []):
            gs_uri = scene.get("gs_uri")
            if gs_uri:
                fresh = get_fresh_signed_url(gs_uri)
                if fresh:
                    scene["image_url"] = fresh

    # Re-sign concept images
    for img in output_data.get("images", []):
        gs_uri = img.get("gs_uri")
        if gs_uri:
            fresh = get_fresh_signed_url(gs_uri)
            if fresh:
                img["image_url"] = fresh

    return output_data


def _node_helper(doc) -> dict:
    if not doc:
        return {}

    stage_key = doc.get("stage_key")
    output_data = doc.get("output_data", {})

    # Re-sign GCS URLs for stages that contain media
    if stage_key in ("video_generation", "storyboard", "image_generation") and output_data:
        import copy
        output_data = _refresh_media_urls(copy.deepcopy(output_data))

    return {
        "_id": str(doc["_id"]),
        "workflow_id": doc.get("workflow_id"),
        "stage_key": stage_key,
        "stage_index": doc.get("stage_index"),
        "stage_type": doc.get("stage_type"),
        "status": doc.get("status"),
        "input_data": doc.get("input_data", {}),
        "output_data": output_data,
        "error": doc.get("error"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_content_workflow(body: ContentWorkflowCreate, request: Request):
    """Create a new content generation workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(body.brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        verify_org_membership(db, brand["organization_id"], workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        workflow = ContentOrchestrator.create_workflow(
            brand_id=body.brand_id,
            title=body.title,
            description=body.description,
            created_by=workos_user_id,
            config=body.config,
        )
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_content_workflows(request: Request, brand_id: Optional[str] = None,
                                  organization_id: Optional[str] = None):
    """List content workflows."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        query = {}
        if brand_id:
            brand = db.brands.find_one({"_id": ObjectId(brand_id)})
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")
            from src.role_helpers import verify_org_membership
            verify_org_membership(db, brand["organization_id"], workos_user_id)
            query["brand_id"] = brand_id
        elif organization_id:
            from src.role_helpers import verify_org_membership
            verify_org_membership(db, organization_id, workos_user_id)
            query["organization_id"] = organization_id
        else:
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user:
                return []
            org_ids = [o["_id"] for o in user.get("organizations", [])]
            if "admin" not in user.get("roles", []):
                query["organization_id"] = {"$in": org_ids}

        workflows = list(db.content_workflows.find(query).sort("created_at", -1))
        return [_workflow_helper(w) for w in workflows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_content_workflow(workflow_id: str, request: Request):
    """Get a single content workflow with its nodes."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        result = _workflow_helper(workflow)
        nodes = list(db.content_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        result["nodes"] = [_node_helper(n) for n in nodes]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_content_workflow(workflow_id: str, body: ContentWorkflowUpdate, request: Request):
    """Update a content workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Deep-merge config so stage_settings updates don't nuke other config keys
        if "config" in update_dict and workflow.get("config"):
            merged_config = {**workflow["config"], **update_dict["config"]}
            update_dict["config"] = merged_config

        update_dict["updated_at"] = datetime.utcnow()
        db.content_workflows.update_one({"_id": ObjectId(workflow_id)}, {"$set": update_dict})

        updated = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        return _workflow_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}", status_code=204)
async def delete_content_workflow(workflow_id: str, request: Request):
    """Delete a content workflow and all associated data."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id, required_roles=["owner"])

        db.content_workflows.delete_one({"_id": ObjectId(workflow_id)})
        db.content_workflow_nodes.delete_many({"workflow_id": workflow_id})
        db.content_workflow_transitions.delete_many({"workflow_id": workflow_id})
        db.content_workflow_states.delete_many({"workflow_id": workflow_id})
        db.content_agent_states.delete_many({"workflow_id": workflow_id})
        db.content_user_sessions.delete_many({"workflow_id": workflow_id})
        db.content_timeline_entries.delete_many({"workflow_id": workflow_id})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Pipeline Control ---

@router.get("/{workflow_id}/nodes")
async def get_nodes(workflow_id: str, request: Request):
    """Get all pipeline nodes for a workflow."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        nodes = list(db.content_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        return [_node_helper(n) for n in nodes]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/run")
async def run_pipeline(workflow_id: str, request: Request):
    """Run the pipeline from the current stage (auto-advances through agent stages)."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.run_pipeline(actor_id=workos_user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/advance")
async def advance_stage(workflow_id: str, request: Request):
    """Advance one stage in the pipeline."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.advance(actor_id=workos_user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_key}/approve")
async def approve_stage(workflow_id: str, stage_key: str, body: ApprovalRequest, request: Request):
    """Approve or reject a human review stage."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)

        if body.approved:
            result = await orchestrator.approve(stage_key, workos_user_id, body.note)
        else:
            result = await orchestrator.reject(stage_key, workos_user_id, body.note)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_key}/input")
async def submit_stage_input(workflow_id: str, stage_key: str, body: HumanInputRequest, request: Request):
    """Submit human input for a stage."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.submit_human_input(stage_key, workos_user_id, body.input_data)
        # When a stage is completed, make it the current stage
        if stage_key in STAGE_ORDER:
            idx = STAGE_ORDER.index(stage_key)
            db.content_workflows.update_one(
                {"_id": ObjectId(workflow_id)},
                {"$set": {"current_stage": stage_key, "current_stage_index": idx}}
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


STAGE_ORDER = [
    "brand", "campaign_strategy", "research", "scheduling",
    "concepts", "image_generation", "storyboard", "video_generation",
    "simulation_testing", "predictive_modeling", "content_ranking",
    "fdm_review", "brand_qa", "publish", "analytics",
]

# Stages that are available (not "coming soon")
AVAILABLE_STAGES = {
    "brand", "campaign_strategy", "scheduling", "concepts",
    "image_generation", "storyboard", "video_generation",
    "simulation_testing", "predictive_modeling", "content_ranking",
    "fdm_review", "brand_qa",
}


def _recalculate_current_stage(workflow_id: str):
    """Set current_stage to the first non-completed available stage, or the last completed one if all done."""
    nodes = {n["stage_key"]: n["status"] for n in db.content_workflow_nodes.find({"workflow_id": workflow_id})}
    for i, key in enumerate(STAGE_ORDER):
        if nodes.get(key) != "completed":
            if key in AVAILABLE_STAGES:
                db.content_workflows.update_one(
                    {"_id": ObjectId(workflow_id)},
                    {"$set": {"current_stage": key, "current_stage_index": i}}
                )
                return
            # Skip unavailable stages, keep looking
            continue
    # All available stages completed — set to last stage
    db.content_workflows.update_one(
        {"_id": ObjectId(workflow_id)},
        {"$set": {"current_stage": STAGE_ORDER[-1], "current_stage_index": len(STAGE_ORDER) - 1}}
    )


@router.post("/{workflow_id}/stages/{stage_key}/reset")
async def reset_stage(workflow_id: str, stage_key: str, request: Request):
    """Reset a completed stage back to pending."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        result = db.content_workflow_nodes.update_one(
            {"workflow_id": workflow_id, "stage_key": stage_key},
            {"$set": {"status": "pending"}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Stage not found")
        _recalculate_current_stage(workflow_id)
        return {"status": "ok", "stage_key": stage_key}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/chat")
async def send_chat_message(workflow_id: str, body: ChatMessageRequest, request: Request):
    """Send a chat message in the workflow context."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.handle_chat_message(workos_user_id, body.message, body.role)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/state")
async def get_workflow_state(workflow_id: str, request: Request):
    """Get the full workflow state snapshot."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.state import WorkflowStateStore
        state = WorkflowStateStore.load(workflow_id)
        return state or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/agent-states")
async def get_agent_states(workflow_id: str, request: Request):
    """Get all agent states for a workflow."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.state import AgentStateStore
        agents = AgentStateStore.list_agents(workflow_id)
        return agents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/transitions")
async def get_transitions(workflow_id: str, request: Request):
    """Get the transition audit trail."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.state import WorkflowState
        state = WorkflowState(workflow_id)
        return state.get_transition_history()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/sessions")
async def get_user_sessions(workflow_id: str, request: Request):
    """Get all active user sessions for a workflow."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.content_generation.state import UserSession
        sessions = UserSession.get_active_sessions(workflow_id)
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Concept Generation ---

@router.post("/{workflow_id}/generate-concepts")
async def generate_concepts(workflow_id: str, body: GenerateConceptsRequest, request: Request):
    """Use AI to generate content concepts with full workflow context."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        # Gather context
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow.get("brand_id") else None
        campaign = None
        strategy = None
        audience = None

        config = workflow.get("config") or {}
        campaign_id = config.get("campaign_id")
        if campaign_id:
            campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
            # Get first strategy for this campaign
            strategy = db.strategies.find_one({"campaign_id": campaign_id})

        if brand:
            audience = db.audiences.find_one({"brand_id": workflow["brand_id"]})

        # Build context string
        context_parts = []
        if brand:
            context_parts.append(f"Brand: {brand.get('name', '')} — {brand.get('industry', '')}. {brand.get('description', '')}. Product: {brand.get('product', '')}")
        if campaign:
            context_parts.append(f"Campaign: {campaign.get('name', '')}. Goal: {campaign.get('campaign_goal', '')}. Platform: {campaign.get('platform', '')}")
        if strategy:
            context_parts.append(f"Strategy: {strategy.get('name', '')}. Budget: {strategy.get('budget_amount', '')} {strategy.get('budget_type', '')}. Objectives: {strategy.get('objectives', '')}")
        if audience:
            context_parts.append(f"Audience: {audience.get('name', '')}. Demographics: {audience.get('demographics', '')}. Interests: {audience.get('interests', '')}")

        # Pull rich research data from workflow config
        research = config.get("stage_settings", {}).get("research", {})
        if research:
            context_parts.append(_build_research_context(research))

        # Check prior stage outputs from workflow state
        try:
            from src.content_generation.state import WorkflowStateStore
            state = WorkflowStateStore.load(workflow_id)
            if state:
                if state.get("strategy_assets"):
                    context_parts.append(f"Strategy assets: {str(state['strategy_assets'])[:500]}")
        except Exception:
            pass

        context_str = "\n".join(context_parts) if context_parts else "No additional context available."

        # Build output schema based on content_type
        content_type = body.content_type or "reel"
        if content_type == "reel":
            schema_desc = """Each concept must have:
- "title": a catchy concept title
- "hook": the opening hook (1-2 sentences that grab attention)
- "script": an array of 3-5 script bullet points (as a JSON array of strings)
- "audio_cues": suggested background music/sound effects
- "duration": suggested duration (e.g. "30s", "60s", "90s")"""
        elif content_type == "carousel":
            schema_desc = """Each concept must have:
- "title": a catchy concept title
- "slides": an array of 3-10 slide objects, each with "image_description" and "caption"
- "messaging": an array of 2-3 key messaging points"""
        elif content_type == "post":
            schema_desc = """Each concept must have:
- "title": a catchy concept title
- "image_description": detailed description of the post image
- "caption": the post caption text
- "messaging": an array of 2-3 key messaging points"""
        elif content_type == "story":
            schema_desc = """Each concept must have:
- "title": a catchy concept title
- "frame_description": description of the story visual frame
- "caption": overlay text for the story
- "cta": call-to-action text (e.g. "Swipe up", "Link in bio")"""
        else:
            schema_desc = """Each concept must have:
- "title": a catchy concept title
- "hook": the opening hook (1-2 sentences)
- "script": a script outline (3-5 bullet points)
- "messaging": an array of 2-3 key messaging points"""

        system_prompt = f"""You are a creative content strategist. Generate {content_type} content concepts based on the following context.

{context_str}

Return a JSON array of exactly {body.num} concepts with tone "{body.tone}" for {content_type} format.
{schema_desc}

Return ONLY valid JSON: {{"concepts": [...]}}"""

        import anthropic
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="Anthropic API key not configured")

        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Generate {body.num} content concepts with a {body.tone} tone."}],
        )

        assistant_text = response.content[0].text

        # Track usage
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        from src.chat import save_llm_usage
        save_llm_usage(
            user_id=workos_user_id,
            campaign_id=campaign_id,
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mode="concept_generation",
        )

        # Parse JSON from response
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', assistant_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"concepts": []}

        # Tag each concept with the tone and content_type
        for concept in result.get("concepts", []):
            concept["tone"] = body.tone
            concept["content_type"] = content_type

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Image Models ---

@router.get("/models/image")
async def get_image_models(request: Request):
    """Get available image generation models."""
    try:
        from src.auth import require_user_id
        require_user_id(request)

        from src.development.models import get_available_image_models
        return {"models": get_available_image_models()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Storyboard Generation ---

@router.post("/{workflow_id}/generate-storyboard")
async def generate_storyboard(workflow_id: str, body: GenerateStoryboardRequest, request: Request):
    """Generate a storyboard (storyline, characters, scenes) from a concept."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        # Look up concept from saved stage outputs
        from src.content_generation.state import WorkflowStateStore
        state_data = WorkflowStateStore.get_state_data(workflow_id)
        concepts_output = state_data.get("stage_outputs", {}).get("concepts", {})
        concepts_list = concepts_output.get("concepts", [])

        # Also check stageSettings saved concepts (per-piece first, then top-level fallback)
        config = workflow.get("config") or {}
        stage_settings = config.get("stage_settings", {})
        concepts_settings = stage_settings.get("concepts", {})
        # Try all pieces to find concepts
        if not concepts_list:
            for piece_data in concepts_settings.get("pieces", {}).values():
                piece_concepts = piece_data.get("generated_concepts", [])
                if piece_concepts:
                    concepts_list = piece_concepts
                    break
        if not concepts_list:
            generated_concepts = concepts_settings.get("generated_concepts", [])
            if generated_concepts:
                concepts_list = generated_concepts

        if body.concept_index < 0 or body.concept_index >= len(concepts_list):
            raise HTTPException(status_code=400, detail=f"Invalid concept_index {body.concept_index}. Available: 0-{len(concepts_list)-1}")

        concept = concepts_list[body.concept_index]
        concept_title = concept.get("title", f"Concept {body.concept_index + 1}")

        # Gather brand context
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow.get("brand_id") else None
        brand_context = ""
        if brand:
            brand_context = f"Brand: {brand.get('name', '')} — {brand.get('industry', '')}. {brand.get('description', '')}. Product: {brand.get('product', '')}"

        # Gather research visual style context
        research_data = stage_settings.get("research", {})
        research_visual_context = _build_visual_style_context(research_data) if research_data else ""

        system_prompt = f"""You are a creative director specializing in short-form video content. Generate a detailed storyboard for this concept.

{brand_context}

{research_visual_context}

Concept: {json.dumps(concept)}

Create a detailed storyboard that breaks this concept into filmable scenes. You must:
1. First define all CHARACTERS with detailed, consistent visual descriptions. Each character's description must be specific enough to generate consistent AI images (age, ethnicity, build, hair, clothing, etc.)
2. Then create SCENES that reference these characters. Each scene's image_prompt should incorporate the character's visual descriptions for consistency.
3. Give each scene a shot type (close-up, medium, wide, over-shoulder, etc.) and duration hint.
4. Use the visual style insights from research (colors, textures, lighting, shot types, objects) to inform your image_prompt descriptions so they match the proven aesthetic of top-performing content in this niche.

Return ONLY valid JSON in this exact format:
{{
  "storyline": "A 1-2 sentence narrative summary",
  "total_cuts": <number of scenes>,
  "characters": [
    {{
      "id": "char_0",
      "name": "Character name",
      "description": "Detailed visual description for consistency across scenes",
      "image_prompt": "Portrait photo prompt optimized for AI image generation"
    }}
  ],
  "scenes": [
    {{
      "id": "scene_0",
      "scene_number": 1,
      "title": "Scene title",
      "description": "What happens in this scene",
      "shot_type": "close-up",
      "duration_hint": "3s",
      "character_ids": ["char_0"],
      "image_prompt": "Detailed image generation prompt incorporating character descriptions"
    }}
  ]
}}"""

        llm_model = body.llm_model or "gemini-pro-3"
        user_prompt = f"Generate a detailed storyboard for the concept: {concept_title}"
        campaign_id = config.get("campaign_id")

        if llm_model.startswith("gemini"):
            # Use Google Gemini
            import google.generativeai as genai
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key not configured")
            genai.configure(api_key=GOOGLE_API_KEY)

            # Map model shorthand to actual Gemini model ID
            gemini_model_map = {
                "gemini-pro-3": "gemini-3-pro-preview",
                "gemini-flash-3": "gemini-3-flash-preview",
                "gemini-flash": "gemini-3-flash-preview",
                "gemini-pro": "gemini-3-pro-preview",
            }
            gemini_model_id = gemini_model_map.get(llm_model, llm_model)

            model = genai.GenerativeModel(gemini_model_id)
            gemini_messages = [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "I understand. I will generate the storyboard as requested."}]},
                {"role": "user", "parts": [{"text": user_prompt}]},
            ]
            response = model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4096,
                    temperature=0.7
                )
            )
            assistant_text = response.text
            input_tokens = getattr(response, 'usage_metadata', None)
            input_tokens = input_tokens.prompt_token_count if input_tokens and hasattr(input_tokens, 'prompt_token_count') else 0
            output_tokens = getattr(response, 'usage_metadata', None)
            output_tokens = output_tokens.candidates_token_count if output_tokens and hasattr(output_tokens, 'candidates_token_count') else 0

            from src.chat import save_llm_usage
            save_llm_usage(
                user_id=workos_user_id,
                campaign_id=campaign_id,
                provider="google",
                model_name=gemini_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                mode="storyboard_generation",
            )

        elif llm_model.startswith("gpt"):
            # Use OpenAI
            from openai import OpenAI as OpenAIClient
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            openai_model_map = {
                "gpt-4o": "gpt-4o",
                "gpt-5.2": "gpt-4o",
            }
            openai_model_id = openai_model_map.get(llm_model, llm_model)

            openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
            response = openai_client.chat.completions.create(
                model=openai_model_id,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            assistant_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            from src.chat import save_llm_usage
            save_llm_usage(
                user_id=workos_user_id,
                campaign_id=campaign_id,
                provider="openai",
                model_name=openai_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                mode="storyboard_generation",
            )

        else:
            # Default: Anthropic Claude
            import anthropic
            ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key not configured")

            claude_model_map = {
                "claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
                "claude-sonnet": "claude-sonnet-4-5-20250929",
                "claude-opus": "claude-opus-4-6",
            }
            claude_model_id = claude_model_map.get(llm_model, "claude-sonnet-4-5-20250929")

            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = anthropic_client.messages.create(
                model=claude_model_id,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            assistant_text = response.content[0].text
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            from src.chat import save_llm_usage
            save_llm_usage(
                user_id=workos_user_id,
                campaign_id=campaign_id,
                provider="anthropic",
                model_name=claude_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                mode="storyboard_generation",
            )

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', assistant_text)
        if json_match:
            storyboard_data = json.loads(json_match.group())
        else:
            raise HTTPException(status_code=500, detail="Failed to parse storyboard from AI response")

        # Add metadata to each character/scene
        image_model = body.image_model or "google/nano-banana-pro"
        for char in storyboard_data.get("characters", []):
            char["image_url"] = None
            char["gs_uri"] = None
            char["image_model"] = image_model
        for scene in storyboard_data.get("scenes", []):
            scene["image_url"] = None
            scene["gs_uri"] = None
            scene["image_model"] = image_model

        # Build the storyboard entry
        # Determine variation_index for this concept
        storyboard_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "storyboard",
        })
        existing_output = storyboard_node.get("output_data", {}) if storyboard_node else {}
        existing_storyboards = existing_output.get("storyboards", [])

        # Count existing variations for this concept to assign next variation_index
        existing_for_concept = [sb for sb in existing_storyboards if sb.get("concept_index") == body.concept_index]
        variation_index = len(existing_for_concept)

        storyboard_entry = {
            "concept_index": body.concept_index,
            "variation_index": variation_index,
            "concept_title": concept_title,
            "storyline": storyboard_data.get("storyline", ""),
            "total_cuts": storyboard_data.get("total_cuts", len(storyboard_data.get("scenes", []))),
            "characters": storyboard_data.get("characters", []),
            "scenes": storyboard_data.get("scenes", []),
            "status": "storyline_ready",
        }

        # Always append as a new variation
        existing_storyboards.append(storyboard_entry)

        output_data = {
            "storyboards": existing_storyboards,
            "status": "storyboard_ready",
        }

        # Save to node output_data
        from src.content_generation.state import WorkflowState
        ws = WorkflowState(workflow_id)
        ws.update_node("storyboard", "completed", output_data=output_data)

        # Save to WorkflowStateStore
        state_data.setdefault("stage_outputs", {})["storyboard"] = output_data
        WorkflowStateStore.save(workflow_id, state_data)

        return storyboard_entry

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateSceneRequest(BaseModel):
    storyboard_index: int = 0
    scene_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    shot_type: Optional[str] = None
    duration_hint: Optional[str] = None
    image_prompt: Optional[str] = None


@router.patch("/{workflow_id}/storyboard-scene")
async def update_storyboard_scene(
    workflow_id: str,
    body: UpdateSceneRequest,
    request: Request,
):
    """Update a single scene's editable fields in a storyboard."""
    try:
        from src.auth import get_workos_user_id
        workos_user_id = get_workos_user_id(request)
        if workos_user_id:
            _verify_workflow_access(workflow_id, workos_user_id)
        else:
            workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

        storyboard_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "storyboard",
        })
        if not storyboard_node or not storyboard_node.get("output_data"):
            raise HTTPException(status_code=400, detail="No storyboard found.")

        output_data = storyboard_node["output_data"]
        storyboards = output_data.get("storyboards", [])

        if body.storyboard_index < 0 or body.storyboard_index >= len(storyboards):
            raise HTTPException(status_code=400, detail=f"Invalid storyboard_index {body.storyboard_index}")

        storyboard = storyboards[body.storyboard_index]
        scenes = storyboard.get("scenes", [])

        scene = next((s for s in scenes if s.get("id") == body.scene_id), None)
        if not scene:
            raise HTTPException(status_code=404, detail=f"Scene {body.scene_id} not found")

        # Update only provided fields
        if body.title is not None:
            scene["title"] = body.title
        if body.description is not None:
            scene["description"] = body.description
        if body.shot_type is not None:
            scene["shot_type"] = body.shot_type
        if body.duration_hint is not None:
            scene["duration_hint"] = body.duration_hint
        if body.image_prompt is not None:
            scene["image_prompt"] = body.image_prompt

        db.content_workflow_nodes.update_one(
            {"_id": storyboard_node["_id"]},
            {"$set": {
                "output_data": output_data,
                "updated_at": datetime.utcnow(),
            }},
        )

        return {"ok": True, "scene": scene}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/generate-storyboard-image")
async def generate_storyboard_image(
    workflow_id: str,
    body: GenerateStoryboardImageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Kick off async image generation for a storyboard character or scene."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)
        config = workflow.get("config", {})
        stage_settings = config.get("stage_settings", {})

        # Load storyboard from node output_data
        storyboard_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "storyboard",
        })
        if not storyboard_node or not storyboard_node.get("output_data"):
            raise HTTPException(status_code=400, detail="No storyboard found. Generate a storyboard first.")

        output_data = storyboard_node["output_data"]
        storyboards = output_data.get("storyboards", [])

        # Find the storyboard for this concept_index + variation_index
        storyboard = None
        concept_matches = [sb for sb in storyboards if sb.get("concept_index") == body.concept_index]
        if body.variation_index < len(concept_matches):
            storyboard = concept_matches[body.variation_index]
        elif concept_matches:
            storyboard = concept_matches[0]
        if not storyboard:
            raise HTTPException(status_code=400, detail=f"No storyboard for concept_index {body.concept_index}")

        # Find the target (character or scene)
        target = None
        image_prompt = None
        if body.target_type == "character":
            for char in storyboard.get("characters", []):
                if char.get("id") == body.target_id:
                    target = char
                    image_prompt = char.get("image_prompt", "")
                    break
        elif body.target_type == "scene":
            for scene in storyboard.get("scenes", []):
                if scene.get("id") == body.target_id:
                    target = scene
                    # Character-first enforcement: check all characters in scene have images
                    char_ids = scene.get("character_ids", [])
                    char_map = {c["id"]: c for c in storyboard.get("characters", [])}
                    for cid in char_ids:
                        char = char_map.get(cid)
                        if char and not char.get("image_url"):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Generate character '{char.get('name', cid)}' image first for visual consistency."
                            )

                    # Enrich scene prompt with character visual descriptions
                    char_descriptions = []
                    for cid in char_ids:
                        char = char_map.get(cid)
                        if char:
                            char_descriptions.append(f"{char.get('name', '')}: {char.get('description', '')}")

                    base_prompt = scene.get("image_prompt", "")
                    if char_descriptions:
                        image_prompt = f"{base_prompt}. Characters in scene: {'; '.join(char_descriptions)}"
                    else:
                        image_prompt = base_prompt
                    break
        else:
            raise HTTPException(status_code=400, detail="target_type must be 'character' or 'scene'")

        if not target:
            raise HTTPException(status_code=404, detail=f"Target {body.target_id} not found in storyboard")

        if not image_prompt:
            raise HTTPException(status_code=400, detail="No image_prompt found for target")

        # Enrich image prompt with visual style from research
        research_data = stage_settings.get("research", {})
        if research_data:
            visual_ctx = _build_visual_style_context(research_data)
            if visual_ctx:
                # Append compact style hint to the image prompt
                style_lines = [l for l in visual_ctx.split("\n")[1:] if l.strip()]  # skip header
                style_hint = ". ".join(style_lines[:3])  # keep compact for image model
                image_prompt = f"{image_prompt}. Style reference: {style_hint}"

        image_model = body.image_model or target.get("image_model", "google/nano-banana")

        # Create background task
        import uuid
        from src.task_manager import task_manager
        task_id = f"storyboard_img_{workflow_id}_{body.target_id}"

        task_manager.create_task(
            task_id,
            "storyboard_image_generation",
            metadata={
                "workflow_id": workflow_id,
                "concept_index": body.concept_index,
                "target_type": body.target_type,
                "target_id": body.target_id,
                "image_model": image_model,
            }
        )

        background_tasks.add_task(
            _generate_storyboard_image_background,
            task_id,
            workflow_id,
            body.concept_index,
            body.variation_index,
            body.target_type,
            body.target_id,
            image_prompt,
            image_model,
        )

        return {"task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/storyboard-image-status/{task_id}")
async def get_storyboard_image_status(workflow_id: str, task_id: str, request: Request):
    """Poll image generation status."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.task_manager import task_manager
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return task

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_storyboard_image_background(
    task_id: str,
    workflow_id: str,
    concept_index: int,
    variation_index: int,
    target_type: str,
    target_id: str,
    image_prompt: str,
    image_model: str,
):
    """Background task: call Replicate API, upload to GCS, update node."""
    import httpx
    import asyncio
    from src.task_manager import task_manager, TaskStatus

    try:
        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10, message="Starting image generation...")

        REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
        if not REPLICATE_API_TOKEN:
            raise Exception("REPLICATE_API_TOKEN not configured")

        # Map model shorthand to Replicate model string
        model_string = image_model
        if "/" not in image_model:
            model_map = {
                "nano-banana": "google/nano-banana",
                "nano-banana-pro": "google/nano-banana-pro",
                "flux-schnell": "black-forest-labs/flux-schnell",
                "flux-pro": "black-forest-labs/flux-1.1-pro",
                "flux-dev": "black-forest-labs/flux-dev",
                "sdxl": "stability-ai/sdxl",
            }
            model_string = model_map.get(image_model, f"google/{image_model}")

        # Call Replicate API
        async with httpx.AsyncClient(timeout=120) as client:
            # Create prediction
            create_resp = await client.post(
                f"https://api.replicate.com/v1/models/{model_string}/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"input": {"prompt": image_prompt}},
            )
            if create_resp.status_code != 201:
                raise Exception(f"Replicate API error: {create_resp.status_code} {create_resp.text}")

            prediction = create_resp.json()
            prediction_url = prediction.get("urls", {}).get("get")
            if not prediction_url:
                raise Exception("No prediction URL returned from Replicate")

            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=30, message="Generating image...")

            # Poll for completion
            for _ in range(120):  # Max 4 minutes
                await asyncio.sleep(2)
                poll_resp = await client.get(
                    prediction_url,
                    headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                )
                poll_data = poll_resp.json()
                status = poll_data.get("status")

                if status == "succeeded":
                    output = poll_data.get("output")
                    if isinstance(output, list) and len(output) > 0:
                        image_url_remote = output[0]
                    elif isinstance(output, str):
                        image_url_remote = output
                    else:
                        raise Exception(f"Unexpected Replicate output format: {output}")
                    break
                elif status == "failed":
                    error = poll_data.get("error", "Unknown error")
                    raise Exception(f"Replicate prediction failed: {error}")
            else:
                raise Exception("Image generation timed out")

            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=60, message="Uploading to storage...")

            # Download image from Replicate's temporary URL
            img_resp = await client.get(image_url_remote)
            if img_resp.status_code != 200:
                raise Exception(f"Failed to download image from Replicate: {img_resp.status_code}")
            image_bytes = img_resp.content

        # Upload to GCS
        from google.cloud import storage as gcs_storage
        from google.oauth2 import service_account

        GCS_BUCKET = os.getenv('GCS_BUCKET', 'video-marketing-simulation')
        service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        if not service_account_json:
            raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

        credentials_dict = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        storage_client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = storage_client.bucket(GCS_BUCKET)

        blob_path = f"storyboards/{workflow_id}/{target_id}.png"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(image_bytes, content_type="image/png")

        gs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET",
            credentials=credentials,
        )

        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=80, message="Updating storyboard...")

        # Update the storyboard node output_data
        storyboard_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "storyboard",
        })
        if storyboard_node:
            output = storyboard_node.get("output_data", {})
            storyboards = output.get("storyboards", [])
            concept_matches = [sb for sb in storyboards if sb.get("concept_index") == concept_index]
            sb = concept_matches[variation_index] if variation_index < len(concept_matches) else (concept_matches[0] if concept_matches else None)
            if sb:
                collection = sb.get("characters" if target_type == "character" else "scenes", [])
                for item in collection:
                    if item.get("id") == target_id:
                        item["image_url"] = signed_url
                        item["gs_uri"] = gs_uri
                        item["image_model"] = image_model
                        break

                # Check if all images are done for this storyboard
                all_chars_done = all(c.get("image_url") for c in sb.get("characters", []))
                all_scenes_done = all(s.get("image_url") for s in sb.get("scenes", []))
                if all_chars_done and all_scenes_done:
                    sb["status"] = "complete"
                elif any(c.get("image_url") for c in sb.get("characters", [])) or any(s.get("image_url") for s in sb.get("scenes", [])):
                    sb["status"] = "images_generating"

            # Save updated node
            db.content_workflow_nodes.update_one(
                {"_id": storyboard_node["_id"]},
                {"$set": {"output_data": output, "updated_at": datetime.utcnow()}},
            )

            # Also update WorkflowStateStore
            from src.content_generation.state import WorkflowStateStore
            state_data = WorkflowStateStore.get_state_data(workflow_id)
            state_data.setdefault("stage_outputs", {})["storyboard"] = output
            WorkflowStateStore.save(workflow_id, state_data)

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Image generated successfully",
            result={"image_url": signed_url, "gs_uri": gs_uri},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=str(e),
            error=str(e),
        )


# --- Concept Image Generation ---

@router.post("/{workflow_id}/generate-concept-image")
async def generate_concept_image(
    workflow_id: str,
    body: GenerateConceptImageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Kick off async image generation for a concept (before storyboard)."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        # Load concepts from workflow config
        config = workflow.get("config", {})
        stage_settings = config.get("stage_settings", {})
        piece_key = body.content_piece_key

        concepts = []
        if piece_key:
            pieces = stage_settings.get("concepts", {}).get("pieces", {})
            piece_data = pieces.get(piece_key, {})
            concepts = piece_data.get("generated_concepts", [])

        # Fallback: try node output
        if not concepts:
            concept_node = db.content_workflow_nodes.find_one({
                "workflow_id": workflow_id,
                "stage_key": "concepts",
            })
            if concept_node and concept_node.get("output_data"):
                concepts = concept_node["output_data"].get("concepts", [])

        if body.concept_index < 0 or body.concept_index >= len(concepts):
            raise HTTPException(status_code=400, detail=f"Invalid concept_index {body.concept_index}. Available: 0-{len(concepts)-1}")

        concept = concepts[body.concept_index]
        content_type = concept.get("content_type", "")

        # Extract image prompt based on content type
        image_prompt = None
        if content_type == "post":
            image_prompt = concept.get("image_description")
        elif content_type == "carousel":
            slides = concept.get("slides", [])
            if body.slide_index is not None:
                if body.slide_index < 0 or body.slide_index >= len(slides):
                    raise HTTPException(status_code=400, detail=f"Invalid slide_index {body.slide_index}. Available: 0-{len(slides)-1}")
                image_prompt = slides[body.slide_index].get("image_description")
            else:
                raise HTTPException(status_code=400, detail="slide_index is required for carousel concepts")
        elif content_type == "story":
            image_prompt = concept.get("frame_description")
        else:
            # Fallback
            image_prompt = concept.get("image_description") or concept.get("hook", "")

        if not image_prompt:
            raise HTTPException(status_code=400, detail="No image description found for this concept")

        # Enrich image prompt with visual style from research
        research_data = stage_settings.get("research", {})
        if research_data:
            visual_ctx = _build_visual_style_context(research_data)
            if visual_ctx:
                style_lines = [l for l in visual_ctx.split("\n")[1:] if l.strip()]
                style_hint = ". ".join(style_lines[:3])
                image_prompt = f"{image_prompt}. Style reference: {style_hint}"

        image_model = body.image_model or "google/nano-banana"

        # Create background task
        from src.task_manager import task_manager
        slide_suffix = body.slide_index if body.slide_index is not None else 0
        task_id = f"concept_img_{workflow_id}_{body.concept_index}_{slide_suffix}"

        task_manager.create_task(
            task_id,
            "concept_image_generation",
            metadata={
                "workflow_id": workflow_id,
                "concept_index": body.concept_index,
                "slide_index": body.slide_index,
                "image_model": image_model,
                "content_piece_key": piece_key,
            }
        )

        background_tasks.add_task(
            _generate_concept_image_background,
            task_id,
            workflow_id,
            body.concept_index,
            body.slide_index,
            image_prompt,
            image_model,
            piece_key,
        )

        return {"task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_concept_image_background(
    task_id: str,
    workflow_id: str,
    concept_index: int,
    slide_index: Optional[int],
    image_prompt: str,
    image_model: str,
    content_piece_key: Optional[str],
):
    """Background task: call Replicate API, upload to GCS, update node & config."""
    import httpx
    import asyncio
    from src.task_manager import task_manager, TaskStatus

    try:
        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10, message="Starting image generation...")

        REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
        if not REPLICATE_API_TOKEN:
            raise Exception("REPLICATE_API_TOKEN not configured")

        # Map model shorthand to Replicate model string
        model_string = image_model
        if "/" not in image_model:
            model_map = {
                "nano-banana": "google/nano-banana",
                "nano-banana-pro": "google/nano-banana-pro",
                "flux-schnell": "black-forest-labs/flux-schnell",
                "flux-pro": "black-forest-labs/flux-1.1-pro",
                "flux-dev": "black-forest-labs/flux-dev",
                "sdxl": "stability-ai/sdxl",
            }
            model_string = model_map.get(image_model, f"google/{image_model}")

        # Call Replicate API
        async with httpx.AsyncClient(timeout=120) as http_client:
            create_resp = await http_client.post(
                f"https://api.replicate.com/v1/models/{model_string}/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"input": {"prompt": image_prompt}},
            )
            if create_resp.status_code != 201:
                raise Exception(f"Replicate API error: {create_resp.status_code} {create_resp.text}")

            prediction = create_resp.json()
            prediction_url = prediction.get("urls", {}).get("get")
            if not prediction_url:
                raise Exception("No prediction URL returned from Replicate")

            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=30, message="Generating image...")

            # Poll for completion
            image_url_remote = None
            for _ in range(120):  # Max 4 minutes
                await asyncio.sleep(2)
                poll_resp = await http_client.get(
                    prediction_url,
                    headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                )
                poll_data = poll_resp.json()
                status = poll_data.get("status")

                if status == "succeeded":
                    output = poll_data.get("output")
                    if isinstance(output, list) and len(output) > 0:
                        image_url_remote = output[0]
                    elif isinstance(output, str):
                        image_url_remote = output
                    else:
                        raise Exception(f"Unexpected Replicate output format: {output}")
                    break
                elif status == "failed":
                    error = poll_data.get("error", "Unknown error")
                    raise Exception(f"Replicate prediction failed: {error}")
            else:
                raise Exception("Image generation timed out")

            task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=60, message="Uploading to storage...")

            # Download image
            img_resp = await http_client.get(image_url_remote)
            if img_resp.status_code != 200:
                raise Exception(f"Failed to download image from Replicate: {img_resp.status_code}")
            image_bytes = img_resp.content

        # Upload to GCS
        from google.cloud import storage as gcs_storage
        from google.oauth2 import service_account

        GCS_BUCKET = os.getenv('GCS_BUCKET', 'video-marketing-simulation')
        service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        if not service_account_json:
            raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

        credentials_dict = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        storage_client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = storage_client.bucket(GCS_BUCKET)

        slide_suffix = slide_index if slide_index is not None else 0
        blob_path = f"concepts/{workflow_id}/{concept_index}_{slide_suffix}.png"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(image_bytes, content_type="image/png")

        gs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET",
            credentials=credentials,
        )

        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=80, message="Saving result...")

        image_record = {
            "concept_index": concept_index,
            "slide_index": slide_index,
            "image_url": signed_url,
            "gs_uri": gs_uri,
            "image_model": image_model,
        }

        # Update image_generation node output_data
        img_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "image_generation",
        })
        if img_node:
            output = img_node.get("output_data", {})
            images = output.get("images", [])
            # Replace existing entry for same concept_index + slide_index, or append
            replaced = False
            for i, existing in enumerate(images):
                if existing.get("concept_index") == concept_index and existing.get("slide_index") == slide_index:
                    images[i] = image_record
                    replaced = True
                    break
            if not replaced:
                images.append(image_record)
            output["images"] = images
            db.content_workflow_nodes.update_one(
                {"_id": img_node["_id"]},
                {"$set": {"output_data": output, "updated_at": datetime.utcnow()}},
            )

        # Also store in workflow config stage_settings
        if content_piece_key:
            workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
            if workflow:
                config = workflow.get("config", {})
                ss = config.get("stage_settings", {})
                img_settings = ss.get("image_generation", {})
                pieces = img_settings.get("pieces", {})
                piece = pieces.get(content_piece_key, {})
                gen_images = piece.get("generated_images", [])
                # Replace or append
                replaced = False
                for i, existing in enumerate(gen_images):
                    if existing.get("concept_index") == concept_index and existing.get("slide_index") == slide_index:
                        gen_images[i] = image_record
                        replaced = True
                        break
                if not replaced:
                    gen_images.append(image_record)
                piece["generated_images"] = gen_images
                pieces[content_piece_key] = piece
                img_settings["pieces"] = pieces
                ss["image_generation"] = img_settings
                config["stage_settings"] = ss
                db.content_workflows.update_one(
                    {"_id": ObjectId(workflow_id)},
                    {"$set": {"config": config, "updated_at": datetime.utcnow()}},
                )

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Image generated successfully",
            result={"image_url": signed_url, "gs_uri": gs_uri},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=str(e),
            error=str(e),
        )


# --- Video Generation ---

@router.post("/{workflow_id}/generate-video")
async def generate_video(
    workflow_id: str,
    body: GenerateVideoRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Generate video(s) from storyboard scenes using the specified model."""
    try:
        from src.auth import get_workos_user_id
        workos_user_id = get_workos_user_id(request)
        if workos_user_id:
            workflow = _verify_workflow_access(workflow_id, workos_user_id)
        else:
            # API key auth — load workflow directly
            workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

        # Load storyboard from node output_data
        storyboard_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "storyboard",
        })
        if not storyboard_node or not storyboard_node.get("output_data"):
            raise HTTPException(status_code=400, detail="No storyboard found. Generate a storyboard first.")

        output_data = storyboard_node["output_data"]
        storyboards = output_data.get("storyboards", [])

        if body.storyboard_index < 0 or body.storyboard_index >= len(storyboards):
            raise HTTPException(status_code=400, detail=f"Invalid storyboard_index {body.storyboard_index}. Available: 0-{len(storyboards)-1}")

        storyboard = storyboards[body.storyboard_index]
        scenes = storyboard.get("scenes", [])
        characters = storyboard.get("characters", [])
        storyline = storyboard.get("storyline", "")

        # Create a task for tracking
        import uuid
        task_id = str(uuid.uuid4())

        # Build initial per-set tracking dict
        sets_tracking = {str(i): {"status": "processing", "stitched_url": None} for i in range(body.count)}

        # Store the video generation job
        job = {
            "task_id": task_id,
            "workflow_id": workflow_id,
            "storyboard_index": body.storyboard_index,
            "model": body.model,
            "count": body.count,
            "output_format": body.output_format,
            "resolution": body.resolution,
            "temperature": body.temperature,
            "status": "pending",
            "scenes_count": len(scenes),
            "characters_count": len(characters),
            "sets": sets_tracking,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        db.video_generation_jobs.insert_one(job)

        # Run generation in background
        background_tasks.add_task(
            _run_video_generation,
            workflow_id=workflow_id,
            task_id=task_id,
            storyboard=storyboard,
            model=body.model,
            count=body.count,
            output_format=body.output_format,
            resolution=body.resolution,
            temperature=body.temperature,
            custom_prompt=body.custom_prompt,
        )

        return {"task_id": task_id, "status": "pending", "model": body.model, "count": body.count}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/video-status/{task_id}")
async def get_video_status(workflow_id: str, task_id: str, request: Request):
    """Check status of a video generation job. Polls external APIs for pending videos."""
    import httpx

    try:
        from src.auth import get_workos_user_id
        get_workos_user_id(request)  # validate if JWT present, but allow API key

        job = db.video_generation_jobs.find_one({"task_id": task_id, "workflow_id": workflow_id})
        if not job:
            raise HTTPException(status_code=404, detail="Video generation job not found")

        # If job is still processing, poll external APIs for each pending video
        if job.get("status") == "processing":
            videos = job.get("videos", [])
            updated = False

            async with httpx.AsyncClient(timeout=30) as http:
                for v in videos:
                    # Process completed OpenAI videos that still need GCS upload
                    if v.get("status") == "completed" and v.get("provider") == "openai" and (v.get("video_url") or "").startswith("openai://"):
                        try:
                            from openai import OpenAI as _OpenAI
                            openai_client = _OpenAI()
                            video_id = v.get("external_id")
                            gs_uri, gcs_url = await _download_and_upload_openai_video(
                                openai_client, video_id, workflow_id, task_id, v["index"],
                            )
                            v["video_url"] = gcs_url
                            v["gs_uri"] = gs_uri
                            updated = True
                            _write_scene_variation_to_node(
                                workflow_id, v["scene_number"], gcs_url, job.get("model", "sora-2"), task_id,
                                gs_uri=gs_uri,
                            )
                        except Exception as dl_err:
                            print(f"Failed to download/upload existing OpenAI video {v.get('external_id')}: {dl_err}")

                    if v.get("status") != "pending":
                        continue

                    try:
                        provider = v.get("provider", "replicate")

                        if provider == "openai":
                            # Poll OpenAI Sora video
                            import asyncio
                            from openai import OpenAI as _OpenAI
                            openai_client = _OpenAI()
                            video_id = v.get("external_id")
                            if not video_id:
                                continue
                            loop = asyncio.get_event_loop()
                            oai_video = await loop.run_in_executor(
                                None,
                                lambda vid=video_id: openai_client.videos.retrieve(vid),
                            )
                            ext_status = oai_video.status

                            if ext_status == "completed":
                                # Download from OpenAI, upload to GCS, write variation immediately
                                try:
                                    gs_uri, gcs_url = await _download_and_upload_openai_video(
                                        openai_client, video_id, workflow_id, task_id, v["index"],
                                    )
                                    v["video_url"] = gcs_url
                                    v["gs_uri"] = gs_uri
                                except Exception as dl_err:
                                    print(f"Failed to download/upload OpenAI video {video_id}: {dl_err}")
                                    v["video_url"] = f"openai://videos/{video_id}/content"
                                v["status"] = "completed"
                                updated = True
                                # Write individual scene variation to node immediately
                                _write_scene_variation_to_node(
                                    workflow_id, v["scene_number"], v.get("video_url"), job.get("model", "sora-2"), task_id,
                                    gs_uri=v.get("gs_uri"),
                                )
                            elif ext_status == "failed":
                                v["status"] = "failed"
                                v["error"] = getattr(oai_video, "error", None) or "OpenAI Sora generation failed"
                                updated = True
                            # else: still queued/in_progress, keep polling

                        elif provider == "google":
                            # Poll Google Veo operation
                            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
                            operation_name = v.get("external_id")
                            if not GOOGLE_API_KEY or not operation_name:
                                continue
                            poll_resp = await http.get(
                                f"https://generativelanguage.googleapis.com/v1beta/{operation_name}",
                                headers={"x-goog-api-key": GOOGLE_API_KEY},
                            )
                            poll_data = poll_resp.json()

                            if poll_data.get("done"):
                                response = poll_data.get("response", {})
                                gen_resp = response.get("generateVideoResponse", response)
                                samples = gen_resp.get("generatedSamples", [])
                                video_uri = None
                                if samples:
                                    video_uri = samples[0].get("video", {}).get("uri")
                                if video_uri:
                                    # Upload to GCS for permanent storage
                                    download_url = f"{video_uri}&key={GOOGLE_API_KEY}"
                                    try:
                                        gs_uri, gcs_url = await _upload_video_to_gcs(
                                            http, download_url, workflow_id, task_id, v["index"],
                                        )
                                        v["video_url"] = gcs_url
                                        v["gs_uri"] = gs_uri
                                    except Exception as gcs_err:
                                        print(f"Failed to upload Veo video to GCS: {gcs_err}")
                                        v["video_url"] = download_url
                                    v["status"] = "completed"
                                    updated = True
                                    _write_scene_variation_to_node(
                                        workflow_id, v["scene_number"], v["video_url"], job.get("model", ""), task_id,
                                        gs_uri=v.get("gs_uri"),
                                    )
                                else:
                                    v["status"] = "failed"
                                    v["error"] = "No video URI in response"
                                    updated = True
                            elif poll_data.get("error"):
                                v["status"] = "failed"
                                v["error"] = str(poll_data["error"])
                                updated = True

                        else:
                            # Poll Replicate prediction
                            REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
                            prediction_url = v.get("external_id")
                            if not REPLICATE_API_TOKEN or not prediction_url:
                                continue
                            poll_resp = await http.get(
                                prediction_url,
                                headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                            )
                            poll_data = poll_resp.json()
                            ext_status = poll_data.get("status")

                            if ext_status == "succeeded":
                                output = poll_data.get("output")
                                raw_url = None
                                if isinstance(output, list) and len(output) > 0:
                                    raw_url = output[0]
                                elif isinstance(output, str):
                                    raw_url = output
                                # Upload to GCS for permanent storage
                                if raw_url:
                                    try:
                                        gs_uri, gcs_url = await _upload_video_to_gcs(
                                            http, raw_url, workflow_id, task_id, v["index"],
                                        )
                                        v["video_url"] = gcs_url
                                        v["gs_uri"] = gs_uri
                                    except Exception as gcs_err:
                                        print(f"Failed to upload Replicate video to GCS: {gcs_err}")
                                        v["video_url"] = raw_url
                                v["status"] = "completed"
                                updated = True
                                _write_scene_variation_to_node(
                                    workflow_id, v["scene_number"], v.get("video_url"), job.get("model", ""), task_id,
                                    gs_uri=v.get("gs_uri"),
                                )
                            elif ext_status == "failed":
                                v["status"] = "failed"
                                v["error"] = poll_data.get("error", "Unknown error")
                                updated = True

                    except Exception as poll_err:
                        print(f"Error polling video {v.get('index')}: {poll_err}")

            if updated:
                db.video_generation_jobs.update_one(
                    {"task_id": task_id},
                    {"$set": {
                        "videos": videos,
                        "updated_at": datetime.utcnow(),
                    }},
                )
                job["videos"] = videos

                # Check each set: if all scenes completed, trigger stitching
                import asyncio
                sets = job.get("sets", {})
                for set_idx_str, set_info in sets.items():
                    if set_info.get("status") != "processing":
                        continue
                    set_idx = int(set_idx_str)
                    set_videos = [v for v in videos if v.get("set_index") == set_idx]
                    if not set_videos:
                        continue
                    all_scenes_done = all(v.get("status") in ("completed", "failed") for v in set_videos)
                    any_scene_completed = any(v.get("status") == "completed" for v in set_videos)
                    if all_scenes_done and any_scene_completed:
                        # Mark set as stitching and kick off background stitch
                        sets[set_idx_str]["status"] = "stitching"
                        db.video_generation_jobs.update_one(
                            {"task_id": task_id},
                            {"$set": {f"sets.{set_idx_str}.status": "stitching", "updated_at": datetime.utcnow()}},
                        )
                        asyncio.ensure_future(_stitch_scene_videos(workflow_id, task_id, set_idx, videos, job.get("model", "")))
                    elif all_scenes_done and not any_scene_completed:
                        # All scenes failed for this set
                        sets[set_idx_str]["status"] = "failed"
                        db.video_generation_jobs.update_one(
                            {"task_id": task_id},
                            {"$set": {f"sets.{set_idx_str}.status": "failed", "updated_at": datetime.utcnow()}},
                        )

                job["sets"] = sets

        # Build scene progress per set for the response
        sets_info = job.get("sets", {})
        scene_progress = {}
        for set_idx_str, set_info in sets_info.items():
            set_videos = [v for v in job.get("videos", []) if v.get("set_index") == int(set_idx_str)]
            completed_scenes = sum(1 for v in set_videos if v.get("status") == "completed")
            total_scenes = len(set_videos)
            scene_progress[set_idx_str] = {
                "completed_scenes": completed_scenes,
                "total_scenes": total_scenes,
                "status": set_info.get("status", "processing"),
                "stitched_url": set_info.get("stitched_url"),
            }

        return {
            "task_id": job["task_id"],
            "status": job.get("status", "unknown"),
            "model": job.get("model"),
            "count": job.get("count"),
            "videos": job.get("videos", []),
            "sets": sets_info,
            "scene_progress": scene_progress,
            "error": job.get("error"),
            "created_at": str(job.get("created_at", "")),
            "updated_at": str(job.get("updated_at", "")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}/video-variation/{variation_id}")
async def delete_video_variation(workflow_id: str, variation_id: str, request: Request):
    """Delete a single video variation by its id."""
    try:
        from src.auth import get_workos_user_id
        get_workos_user_id(request)

        result = db.content_workflow_nodes.update_one(
            {"workflow_id": workflow_id, "stage_key": "video_generation"},
            {"$pull": {"output_data.variations": {"id": variation_id}}},
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Variation not found")
        return {"ok": True, "deleted_variation_id": variation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/video-jobs")
async def list_video_jobs(workflow_id: str, request: Request):
    """List all video generation jobs for a workflow."""
    try:
        from src.auth import get_workos_user_id
        get_workos_user_id(request)

        jobs = list(db.video_generation_jobs.find({"workflow_id": workflow_id}).sort("created_at", -1))
        result = []
        for j in jobs:
            videos = j.get("videos", [])
            done = sum(1 for v in videos if v.get("status") == "completed")
            failed = sum(1 for v in videos if v.get("status") == "failed")
            pending = len(videos) - done - failed
            result.append({
                "task_id": j.get("task_id", ""),
                "model": j.get("model", ""),
                "status": j.get("status", ""),
                "temperature": j.get("temperature"),
                "scenes_total": len(videos),
                "scenes_done": done,
                "scenes_failed": failed,
                "scenes_pending": pending,
                "created_at": str(j.get("created_at", "")),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}/video-job/{task_id}", status_code=200)
async def delete_video_job(workflow_id: str, task_id: str, request: Request):
    """Delete a video generation job and its scene variations."""
    try:
        from src.auth import get_workos_user_id
        get_workos_user_id(request)

        job = db.video_generation_jobs.find_one({"task_id": task_id, "workflow_id": workflow_id})
        if not job:
            raise HTTPException(status_code=404, detail="Video generation job not found")

        # Remove the job document
        db.video_generation_jobs.delete_one({"task_id": task_id, "workflow_id": workflow_id})

        # Remove scene variations for this task_id from the node
        db.content_workflow_nodes.update_one(
            {"workflow_id": workflow_id, "stage_key": "video_generation"},
            {"$pull": {"output_data.variations": {"task_id": task_id}}},
        )

        return {"ok": True, "deleted_task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _run_video_generation(
    workflow_id: str,
    task_id: str,
    storyboard: dict,
    model: str,
    count: int,
    output_format: str | None,
    resolution: str | None,
    temperature: float | None,
    custom_prompt: str | None = None,
):
    """Background task: submit video generation jobs to external APIs (non-blocking)."""
    import httpx
    import asyncio

    try:
        db.video_generation_jobs.update_one(
            {"task_id": task_id},
            {"$set": {"status": "running", "updated_at": datetime.utcnow()}},
        )

        # Resolve model config from VIDEO_MODELS registry
        from src.development.models import VIDEO_MODELS
        model_key = model.replace(".", "-")
        model_cfg = VIDEO_MODELS.get(model_key) or VIDEO_MODELS.get(model)

        is_google_direct = model_cfg and getattr(model_cfg, "platform", "") == "direct" and getattr(model_cfg, "provider", "") == "google"
        is_openai_direct = model_cfg and getattr(model_cfg, "platform", "") == "direct" and getattr(model_cfg, "provider", "") == "openai"

        scenes = storyboard.get("scenes", [])
        characters = storyboard.get("characters", [])
        storyline = storyboard.get("storyline", "")
        videos = []
        global_idx = 0

        # Determine provider string
        if is_openai_direct:
            provider_str = "openai"
        elif is_google_direct:
            provider_str = "google"
        else:
            provider_str = "replicate"

        GOOGLE_BATCH_SIZE = 3
        GOOGLE_POLL_INTERVAL = 10  # seconds between poll checks

        async with httpx.AsyncClient(timeout=60) as http:
            for set_idx in range(count):
                # Build all scene entries first
                scene_entries = []
                for j, scene in enumerate(scenes):
                    scene_chars = [c for c in characters if c.get("id") in scene.get("character_ids", [])]
                    char_desc = ", ".join([f"{c['name']}: {c.get('description', '')}" for c in scene_chars])
                    duration_hint = scene.get("duration_hint", "5s")

                    if custom_prompt:
                        # Interpolate user-provided template with scene variables
                        scene_prompt = custom_prompt.replace("{storyline}", storyline)
                        scene_prompt = scene_prompt.replace("{scene_number}", str(scene.get("scene_number", j + 1)))
                        scene_prompt = scene_prompt.replace("{title}", scene.get("title", ""))
                        scene_prompt = scene_prompt.replace("{description}", scene.get("description", ""))
                        scene_prompt = scene_prompt.replace("{characters}", char_desc)
                        scene_prompt = scene_prompt.replace("{shot_type}", scene.get("shot_type", ""))
                        scene_prompt = scene_prompt.replace("{duration_hint}", duration_hint)
                    else:
                        scene_prompt = f"Storyline context: {storyline}\n\n"
                        scene_prompt += f"Generate ONLY this specific scene as a single video clip:\n"
                        scene_prompt += f"Scene {scene.get('scene_number', j + 1)}: {scene.get('title', '')}. {scene.get('description', '')}."
                        if char_desc:
                            scene_prompt += f" Characters: {char_desc}."
                        if scene.get("shot_type"):
                            scene_prompt += f" Shot type: {scene['shot_type']}."
                        scene_prompt += f"\n\nIMPORTANT: This video clip MUST be exactly {duration_hint} long."

                    scene_entries.append({
                        "index": global_idx + j,
                        "set_index": set_idx,
                        "scene_index": j,
                        "scene_number": scene.get("scene_number", j + 1),
                        "model": model,
                        "prompt": scene_prompt,
                        "duration_hint": duration_hint,
                        "status": "pending",
                        "video_url": None,
                        "external_id": None,
                        "provider": provider_str,
                    })

                if is_google_direct:
                    # Submit in batches, wait for each batch to complete before next
                    for batch_start in range(0, len(scene_entries), GOOGLE_BATCH_SIZE):
                        batch = scene_entries[batch_start:batch_start + GOOGLE_BATCH_SIZE]

                        # Submit this batch
                        for entry in batch:
                            try:
                                external_id = await _submit_google_veo(http, entry["prompt"], model_cfg, resolution, output_format)
                                entry["external_id"] = external_id
                            except Exception as submit_err:
                                entry["status"] = "failed"
                                entry["error"] = str(submit_err)
                            await asyncio.sleep(2)  # small gap between submissions within batch

                        # Save progress so UI can see submitted scenes
                        videos.extend(batch)
                        db.video_generation_jobs.update_one(
                            {"task_id": task_id},
                            {"$set": {"videos": videos, "updated_at": datetime.utcnow()}},
                        )

                        # Wait for this batch to complete before submitting next
                        if batch_start + GOOGLE_BATCH_SIZE < len(scene_entries):
                            pending_ids = [e["external_id"] for e in batch if e.get("external_id")]
                            for _ in range(60):  # max ~10 min wait per batch
                                await asyncio.sleep(GOOGLE_POLL_INTERVAL)
                                all_done = True
                                for ext_id in pending_ids:
                                    try:
                                        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
                                        api_base = "https://generativelanguage.googleapis.com/v1beta"
                                        poll_resp = await http.get(
                                            f"{api_base}/{ext_id}",
                                            headers={"x-goog-api-key": GOOGLE_API_KEY},
                                        )
                                        if poll_resp.status_code == 200:
                                            op = poll_resp.json()
                                            if not op.get("done", False):
                                                all_done = False
                                                break
                                    except Exception:
                                        all_done = False
                                        break
                                if all_done:
                                    break
                else:
                    # OpenAI / Replicate: submit all with small delays
                    for entry in scene_entries:
                        try:
                            if is_openai_direct:
                                external_id = await _submit_openai_sora(entry["prompt"], model_cfg, entry["duration_hint"], output_format)
                            else:
                                slug = getattr(model_cfg, "model_string", model) if model_cfg else model
                                external_id = await _submit_replicate(http, entry["prompt"], slug, resolution)
                            entry["external_id"] = external_id
                        except Exception as submit_err:
                            entry["status"] = "failed"
                            entry["error"] = str(submit_err)
                        if is_openai_direct:
                            await asyncio.sleep(2)
                    videos.extend(scene_entries)

                global_idx += len(scene_entries)

        # All per-scene jobs submitted — mark as processing so the status endpoint will poll
        db.video_generation_jobs.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "processing",
                "videos": videos,
                "updated_at": datetime.utcnow(),
            }},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.video_generation_jobs.update_one(
            {"task_id": task_id},
            {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow()}},
        )


async def _submit_openai_sora(prompt: str, model_cfg, duration_hint: str, output_format: str | None = None) -> str:
    """Submit a video generation request to OpenAI Sora. Returns the video ID."""
    import asyncio
    from openai import OpenAI

    openai_client = OpenAI()
    model_string = getattr(model_cfg, "model_string", "sora-2")

    # Map duration_hint to allowed Sora seconds ("4", "8", "12")
    # Scene <=4s → "4", scene >4s and <=8s → "8", scene >8s → "12"
    dur_num = int(''.join(filter(str.isdigit, duration_hint)) or '4')
    if dur_num <= 4:
        sora_seconds = "4"
    elif dur_num <= 8:
        sora_seconds = "8"
    else:
        sora_seconds = "12"

    # Map output_format to Sora size
    FORMAT_TO_SIZE = {
        "reel_9_16": "720x1280",
        "story_9_16": "720x1280",
        "post_1_1": "1024x1024",
        "landscape_16_9": "1280x720",
    }
    size = FORMAT_TO_SIZE.get(output_format, "1280x720")

    # OpenAI SDK is sync — run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    video = await loop.run_in_executor(
        None,
        lambda: openai_client.videos.create(
            model=model_string,
            prompt=prompt,
            size=size,
            seconds=sora_seconds,
        ),
    )

    video_id = video.id
    if not video_id:
        raise Exception(f"No video ID returned from OpenAI Sora: {video}")
    return video_id


async def _submit_google_veo(http, prompt: str, model_cfg, resolution: str | None, output_format: str | None = None) -> str:
    """Submit a video generation request to Google Veo. Returns the operation name."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise Exception("GOOGLE_API_KEY not configured")

    model_string = getattr(model_cfg, "model_string", "veo-3.1-generate-preview")
    api_base = "https://generativelanguage.googleapis.com/v1beta"

    # Map output_format to aspect ratio
    FORMAT_TO_ASPECT = {
        "reel_9_16": "9:16",
        "story_9_16": "9:16",
        "post_1_1": "1:1",
        "landscape_16_9": "16:9",
    }

    params: dict = {"personGeneration": "allow_all"}
    if resolution:
        params["resolution"] = resolution
    if output_format and output_format in FORMAT_TO_ASPECT:
        params["aspectRatio"] = FORMAT_TO_ASPECT[output_format]

    create_resp = await http.post(
        f"{api_base}/models/{model_string}:predictLongRunning",
        headers={
            "x-goog-api-key": GOOGLE_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "instances": [{"prompt": prompt}],
            "parameters": params,
        },
    )
    if create_resp.status_code not in (200, 201):
        raise Exception(f"Google Veo API error: {create_resp.status_code} {create_resp.text}")

    operation = create_resp.json()
    operation_name = operation.get("name")
    if not operation_name:
        raise Exception(f"No operation name returned from Google Veo: {operation}")
    return operation_name


async def _submit_replicate(http, prompt: str, model_slug: str, resolution: str | None) -> str:
    """Submit a video generation request to Replicate. Returns the prediction poll URL."""
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    if not REPLICATE_API_TOKEN:
        raise Exception("REPLICATE_API_TOKEN not configured")

    input_data = {"prompt": prompt}
    if resolution:
        input_data["resolution"] = resolution

    create_resp = await http.post(
        f"https://api.replicate.com/v1/models/{model_slug}/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"input": input_data},
    )
    if create_resp.status_code != 201:
        raise Exception(f"Replicate API error: {create_resp.status_code} {create_resp.text}")

    prediction = create_resp.json()
    prediction_url = prediction.get("urls", {}).get("get")
    if not prediction_url:
        raise Exception("No prediction URL returned from Replicate")
    return prediction_url


async def _download_and_upload_openai_video(openai_client, video_id: str, workflow_id: str, task_id: str, index: int) -> tuple:
    """Download video from OpenAI Sora and upload to GCS. Returns (gs_uri, signed_url)."""
    import asyncio
    import tempfile

    loop = asyncio.get_event_loop()

    # Download from OpenAI
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    await loop.run_in_executor(
        None,
        lambda: openai_client.videos.download_content(video_id).stream_to_file(tmp_path),
    )

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()
    os.unlink(tmp_path)

    # Upload to GCS
    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account

    GCS_BUCKET = os.getenv("GCS_BUCKET", "video-marketing-simulation")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

    credentials_dict = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    storage_client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = storage_client.bucket(GCS_BUCKET)

    blob_path = f"videos/{workflow_id}/{task_id}-scene{index}.mp4"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(video_bytes, content_type="video/mp4")

    gs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=7),
        method="GET",
        credentials=credentials,
    )
    return (gs_uri, signed_url)


def _write_scene_variation_to_node(workflow_id: str, scene_number: int, video_url: str, model: str, task_id: str = "", gs_uri: str = None):
    """Write a single completed scene video as a variation to the video_generation node immediately."""
    if not video_url:
        return

    variation = {
        "id": f"{task_id}-scene{scene_number}-{model}",
        "title": f"Scene {scene_number} — {model}",
        "preview": video_url,
        "type": "scene",
        "task_id": task_id,
        "scene_number": scene_number,
        "model": model,
    }
    if gs_uri:
        variation["gs_uri"] = gs_uri

    video_node = db.content_workflow_nodes.find_one({
        "workflow_id": workflow_id,
        "stage_key": "video_generation",
    })

    if video_node:
        existing_variations = video_node.get("output_data", {}).get("variations", [])
        # Avoid duplicates
        if not any(v.get("id") == variation["id"] for v in existing_variations):
            existing_variations.append(variation)
            db.content_workflow_nodes.update_one(
                {"_id": video_node["_id"]},
                {"$set": {
                    "output_data.variations": existing_variations,
                    "updated_at": datetime.utcnow(),
                }},
            )
    else:
        db.content_workflow_nodes.insert_one({
            "workflow_id": workflow_id,
            "stage_key": "video_generation",
            "status": "in_progress",
            "output_data": {
                "variations": [variation],
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })


async def _upload_video_to_gcs(http, video_uri: str, workflow_id: str, task_id: str, index: int, headers: dict = None) -> tuple:
    """Download video from temporary URL and upload to GCS. Returns (gs_uri, signed_url)."""
    vid_resp = await http.get(video_uri, headers=headers or {}, follow_redirects=True, timeout=120)
    if vid_resp.status_code != 200:
        raise Exception(f"Failed to download video: {vid_resp.status_code}")
    video_bytes = vid_resp.content

    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account

    GCS_BUCKET = os.getenv("GCS_BUCKET", "video-marketing-simulation")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

    credentials_dict = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    storage_client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = storage_client.bucket(GCS_BUCKET)

    blob_path = f"videos/{workflow_id}/{task_id}-{index}.mp4"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(video_bytes, content_type="video/mp4")

    gs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=7),
        method="GET",
        credentials=credentials,
    )
    return (gs_uri, signed_url)


async def _stitch_scene_videos(workflow_id: str, task_id: str, set_idx: int, videos: list, model: str):
    """Download scene clips for a set, concatenate with ffmpeg, upload stitched video to GCS."""
    import tempfile
    import subprocess
    import httpx

    try:
        # Filter videos for this set, sorted by scene_index
        set_videos = sorted(
            [v for v in videos if v.get("set_index") == set_idx and v.get("status") == "completed"],
            key=lambda v: v.get("scene_index", 0),
        )

        if not set_videos:
            db.video_generation_jobs.update_one(
                {"task_id": task_id},
                {"$set": {f"sets.{set_idx}.status": "failed", f"sets.{set_idx}.error": "No completed scenes", "updated_at": datetime.utcnow()}},
            )
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download each scene clip
            clip_paths = []
            async with httpx.AsyncClient(timeout=120) as http:
                for v in set_videos:
                    video_url = v["video_url"]
                    scene_idx = v["scene_index"]
                    clip_path = os.path.join(tmpdir, f"scene_{scene_idx:03d}.mp4")

                    if video_url.startswith("openai://videos/"):
                        # Download from OpenAI Sora API
                        import asyncio
                        from openai import OpenAI as _OpenAI
                        openai_client = _OpenAI()
                        oai_video_id = video_url.replace("openai://videos/", "").replace("/content", "")
                        loop = asyncio.get_event_loop()
                        content_resp = await loop.run_in_executor(
                            None,
                            lambda vid=oai_video_id: openai_client.videos.download_content(vid),
                        )
                        content_resp.stream_to_file(clip_path)
                    else:
                        resp = await http.get(video_url, follow_redirects=True, timeout=120)
                        if resp.status_code != 200:
                            raise Exception(f"Failed to download scene {scene_idx}: HTTP {resp.status_code}")
                        with open(clip_path, "wb") as f:
                            f.write(resp.content)

                    clip_paths.append(clip_path)

            # Write ffmpeg concat list
            list_path = os.path.join(tmpdir, "concat_list.txt")
            with open(list_path, "w") as f:
                for cp in clip_paths:
                    f.write(f"file '{cp}'\n")

            # Run ffmpeg to concatenate
            output_path = os.path.join(tmpdir, "stitched_output.mp4")
            result = subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c:v", "libx264", "-c:a", "aac", "-y", output_path],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise Exception(f"ffmpeg failed: {result.stderr}")

            # Upload stitched video to GCS
            from google.cloud import storage as gcs_storage
            from google.oauth2 import service_account

            GCS_BUCKET = os.getenv("GCS_BUCKET", "video-marketing-simulation")
            service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not service_account_json:
                raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

            credentials_dict = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            storage_client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
            bucket = storage_client.bucket(GCS_BUCKET)

            blob_path = f"videos/{workflow_id}/{task_id}-set{set_idx}-stitched.mp4"
            blob = bucket.blob(blob_path)

            with open(output_path, "rb") as f:
                blob.upload_from_file(f, content_type="video/mp4")

            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=7),
                method="GET",
                credentials=credentials,
            )

        gs_uri = f"gs://{GCS_BUCKET}/{blob_path}"

        # Update job: mark this set as done with stitched URL and gs_uri
        db.video_generation_jobs.update_one(
            {"task_id": task_id},
            {"$set": {
                f"sets.{set_idx}.status": "done",
                f"sets.{set_idx}.stitched_url": signed_url,
                f"sets.{set_idx}.gs_uri": gs_uri,
                "updated_at": datetime.utcnow(),
            }},
        )

        # Check if all sets are done — if so, mark job completed and write variations
        job = db.video_generation_jobs.find_one({"task_id": task_id})
        if job:
            sets = job.get("sets", {})
            all_sets_done = all(s.get("status") in ("done", "failed") for s in sets.values())
            if all_sets_done:
                db.video_generation_jobs.update_one(
                    {"task_id": task_id},
                    {"$set": {"status": "completed", "updated_at": datetime.utcnow()}},
                )
                _write_video_variations_to_node(workflow_id, task_id, job.get("videos", []), sets, model)

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.video_generation_jobs.update_one(
            {"task_id": task_id},
            {"$set": {
                f"sets.{set_idx}.status": "failed",
                f"sets.{set_idx}.error": str(e),
                "updated_at": datetime.utcnow(),
            }},
        )


def _write_video_variations_to_node(workflow_id: str, task_id: str, videos: list, sets: dict, model: str):
    """Write one variation per completed stitched set to the video_generation node's output_data."""
    variations = []
    for set_idx_str, set_info in sorted(sets.items(), key=lambda x: int(x[0])):
        stitched_url = set_info.get("stitched_url")
        if stitched_url:
            set_idx = int(set_idx_str)
            var = {
                "id": f"{task_id}-set{set_idx}",
                "title": f"Video {set_idx + 1} — {model}",
                "preview": stitched_url,
                "type": "video",
                "status": "draft",
            }
            gs_uri = set_info.get("gs_uri")
            if gs_uri:
                var["gs_uri"] = gs_uri
            variations.append(var)

    video_node = db.content_workflow_nodes.find_one({
        "workflow_id": workflow_id,
        "stage_key": "video_generation",
    })
    if video_node:
        existing_videos = video_node.get("output_data", {}).get("videos", [])
        existing_videos.extend(videos)
        existing_variations = video_node.get("output_data", {}).get("variations", [])
        existing_variations.extend(variations)
        db.content_workflow_nodes.update_one(
            {"_id": video_node["_id"]},
            {"$set": {
                "output_data.videos": existing_videos,
                "output_data.variations": existing_variations,
                "status": "completed",
                "updated_at": datetime.utcnow(),
            }},
        )
    else:
        # Create the node if it doesn't exist
        db.content_workflow_nodes.insert_one({
            "workflow_id": workflow_id,
            "stage_key": "video_generation",
            "status": "completed",
            "output_data": {
                "videos": videos,
                "variations": variations,
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })


# --- Simulation ---

class RunSimulationRequest(BaseModel):
    genders: List[str] = Field(default=["Male", "Female"])
    ages: List[str] = Field(default=["18-24", "25-34"])
    model_provider: str = "google"
    model_name: str = "gemini-2.5-flash"
    persona_ids: Optional[List[str]] = None
    video_ids: Optional[List[str]] = None  # specific video variation IDs; None = all


@router.post("/{workflow_id}/simulate")
async def run_simulation(workflow_id: str, body: RunSimulationRequest, request: Request):
    """Run LLM-based simulation scoring for gender × age demographic segments."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        # Gather context from workflow and related data
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow.get("brand_id") else None
        brand_name = brand.get("name", "Unknown Brand") if brand else "Unknown Brand"
        brand_description = brand.get("description", "") if brand else ""

        campaign_id = (workflow.get("config") or {}).get("campaign_id")
        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)}) if campaign_id else None
        campaign_name = campaign.get("name", "") if campaign else ""
        campaign_goal = campaign.get("campaign_goal", "") if campaign else ""

        # Get video/storyboard context from nodes
        nodes = {n["stage_key"]: n for n in db.content_workflow_nodes.find({"workflow_id": workflow_id})}
        concepts_output = nodes.get("concepts", {}).get("output_data", {})
        storyboard_output = nodes.get("storyboard", {}).get("output_data", {})
        video_output = nodes.get("video_generation", {}).get("output_data", {})

        concepts_summary = ""
        for c in concepts_output.get("concepts", [])[:3]:
            concepts_summary += f"- {c.get('title', 'Untitled')}: {c.get('hook', '')}\n"

        storyboard_summary = ""
        for sb in storyboard_output.get("storyboards", [])[:2]:
            storyboard_summary += f"- Storyline: {sb.get('storyline', '')[:200]}\n"
            storyboard_summary += f"  Scenes: {len(sb.get('scenes', []))} cuts\n"

        all_variations = video_output.get("variations", [])
        # Get full videos (stitched + video types) for simulation
        full_videos = [v for v in all_variations if v.get("type") in ("stitched", "video") and v.get("preview")]
        if body.video_ids:
            full_videos = [v for v in full_videos if v.get("id") in body.video_ids]
        # Fallback: if no full videos, use all variations
        if not full_videos:
            full_videos = [{"id": "all", "title": "All Content", "model": "", "type": "all"}]

        # Fetch persona context if persona_ids provided
        persona_context = ""
        if body.persona_ids:
            for pid in body.persona_ids:
                try:
                    persona = db.personas.find_one({"_id": ObjectId(pid)})
                    if persona:
                        p_name = persona.get("name", "Unknown")
                        p_desc = persona.get("description", "")
                        p_demos = persona.get("demographics", {})
                        demo_parts = []
                        for k, v in p_demos.items():
                            if v and k not in ("custom_fields",):
                                if isinstance(v, list) and len(v) > 0:
                                    demo_parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                                elif isinstance(v, (str, int, float)) and v:
                                    demo_parts.append(f"{k}: {v}")
                        persona_context += f"\n- {p_name}: {p_desc}"
                        if demo_parts:
                            persona_context += f" ({'; '.join(demo_parts)})"
                except Exception:
                    pass

        # Build combos list
        combos = []
        for g in body.genders:
            for a in body.ages:
                combos.append({"gender": g, "age": a})
        combos_str = json.dumps(combos)

        # Resolve provider from model name
        llm_model = body.model_name
        provider = body.model_provider
        if not provider:
            if llm_model.startswith("gemini") or llm_model == "gemini-pro-3":
                provider = "google"
            elif llm_model.startswith("gpt") or llm_model.startswith("o1"):
                provider = "openai"
            elif llm_model.startswith("claude"):
                provider = "anthropic"
            else:
                provider = "google"

        model_map = {
            "gemini-pro-3": "gemini-2.5-pro",
            "claude-4.5-sonnet": "claude-sonnet-4.5",
            "gpt-5.2": "gpt-4o",
        }
        mapped_model = model_map.get(llm_model, llm_model)

        from src.ai_agent import evaluate_with_openai, evaluate_with_anthropic, evaluate_with_google

        # Run simulation per video
        all_results = []
        for vid in full_videos:
            vid_id = vid.get("id", "all")
            vid_title = vid.get("title", "All Content")
            vid_model = vid.get("model", "")
            vid_type = vid.get("type", "")

            video_desc = f"{vid_title}"
            if vid_model:
                video_desc += f" (generated with {vid_model})"
            if vid_type == "stitched":
                video_desc += " — full stitched video from all scenes"
            elif vid_type == "video":
                video_desc += " — complete video"

            persona_section = ""
            if persona_context:
                persona_section = "PERSONAS (use these as additional context for scoring):" + persona_context

            prompt = f"""You are an expert marketing analyst. Score how well this specific video would resonate with each demographic segment.

BRAND: {brand_name}
{f"Brand Description: {brand_description}" if brand_description else ""}
{f"Campaign: {campaign_name}" if campaign_name else ""}
{f"Campaign Goal: {campaign_goal}" if campaign_goal else ""}

CONTENT CONTEXT:
Concepts:
{concepts_summary if concepts_summary else "No concepts generated yet."}

Storyboards:
{storyboard_summary if storyboard_summary else "No storyboards generated yet."}

VIDEO BEING EVALUATED: {video_desc}
{persona_section}
DEMOGRAPHIC SEGMENTS TO SCORE:
{combos_str}

For EACH segment, provide a score from 0-100 and a brief reasoning (1-2 sentences).

Return ONLY a JSON object with this exact structure:
{{"results": [
  {{"gender": "...", "age": "...", "score": 0-100, "reasoning": "..."}},
  ...
]}}

Score based on:
- Content relevance to the demographic
- Visual style and tone alignment
- Platform engagement patterns for the age group
- Messaging effectiveness for the gender/age combination
"""

            result = None
            if provider == "openai":
                result = evaluate_with_openai(prompt, mapped_model)
            elif provider == "anthropic":
                result = evaluate_with_anthropic(prompt, mapped_model)
            elif provider == "google":
                result = evaluate_with_google(prompt, mapped_model)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

            if not result:
                continue

            vid_results = result.get("results", [])
            for r in vid_results:
                r["score"] = max(0, min(100, int(r.get("score", 0))))
                r["video_id"] = vid_id
                r["video_title"] = vid_title
            all_results.extend(vid_results)

        results = all_results
        if not results:
            raise HTTPException(status_code=500, detail="LLM returned no results for any video")

        # Save to simulation_testing node output_data
        sim_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "simulation_testing",
        })
        output_data = {
            "results": results,
            "config": {
                "genders": body.genders,
                "ages": body.ages,
                "model_provider": provider,
                "model_name": llm_model,
            },
            "run_at": datetime.utcnow().isoformat(),
        }
        if sim_node:
            db.content_workflow_nodes.update_one(
                {"_id": sim_node["_id"]},
                {"$set": {
                    "output_data": output_data,
                    "status": "completed",
                    "updated_at": datetime.utcnow(),
                }},
            )
        else:
            # Determine stage_index
            stage_index = STAGE_ORDER.index("simulation_testing") if "simulation_testing" in STAGE_ORDER else 7
            db.content_workflow_nodes.insert_one({
                "workflow_id": workflow_id,
                "stage_key": "simulation_testing",
                "stage_index": stage_index,
                "stage_type": "agent",
                "status": "completed",
                "input_data": {},
                "output_data": output_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Predictive Modeling ---


def _compute_research_benchmarks(config: dict) -> dict:
    """Extract avg views/likes/comments/engagement from research data."""
    stage_settings = config.get("stage_settings", {})
    research = stage_settings.get("research", {})

    def _avg_kpis(performers: list) -> dict:
        if not performers:
            return {"avg_views": 0, "avg_likes": 0, "avg_comments": 0, "count": 0}
        total_views = sum(p.get("videoPlayCount", 0) for p in performers)
        total_likes = sum(p.get("likesCount", 0) for p in performers)
        total_comments = sum(p.get("commentsCount", 0) for p in performers)
        n = len(performers)
        return {
            "avg_views": round(total_views / n) if n else 0,
            "avg_likes": round(total_likes / n) if n else 0,
            "avg_comments": round(total_comments / n) if n else 0,
            "count": n,
        }

    benchmarks = {}

    brand_ig = research.get("brand_instagram")
    if brand_ig and isinstance(brand_ig, dict):
        followers = brand_ig.get("followers", 0)
        kpis = _avg_kpis(brand_ig.get("top_performers", []))
        kpis["followers"] = followers
        if followers and kpis["avg_views"]:
            kpis["engagement_rate"] = round(
                (kpis["avg_likes"] + kpis["avg_comments"]) / followers * 100, 2
            )
        else:
            kpis["engagement_rate"] = 0
        benchmarks["brand"] = kpis

    comp_ig = research.get("competitor_instagram", {})
    if isinstance(comp_ig, dict):
        comp_benchmarks = {}
        for comp_user, comp_data in comp_ig.items():
            if isinstance(comp_data, dict) and "error" not in comp_data:
                followers = comp_data.get("followers", 0)
                kpis = _avg_kpis(comp_data.get("top_performers", []))
                kpis["followers"] = followers
                if followers and kpis["avg_views"]:
                    kpis["engagement_rate"] = round(
                        (kpis["avg_likes"] + kpis["avg_comments"]) / followers * 100, 2
                    )
                else:
                    kpis["engagement_rate"] = 0
                comp_benchmarks[comp_user] = kpis
        if comp_benchmarks:
            benchmarks["competitors"] = comp_benchmarks

    return benchmarks


class PredictRequest(BaseModel):
    model_name: str = "gemini-pro-3"
    video_ids: Optional[List[str]] = None


@router.post("/{workflow_id}/predict")
async def run_predictive_modeling(workflow_id: str, body: PredictRequest, request: Request):
    """Run LLM-based predictive modeling to forecast KPIs for each video."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        config = workflow.get("config") or {}

        # Gather context
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow.get("brand_id") else None
        brand_name = brand.get("name", "Unknown Brand") if brand else "Unknown Brand"
        brand_description = brand.get("description", "") if brand else ""

        campaign_id = config.get("campaign_id")
        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)}) if campaign_id else None
        campaign_name = campaign.get("name", "") if campaign else ""
        campaign_goal = campaign.get("campaign_goal", "") if campaign else ""

        nodes = {n["stage_key"]: n for n in db.content_workflow_nodes.find({"workflow_id": workflow_id})}
        concepts_output = nodes.get("concepts", {}).get("output_data", {})
        storyboard_output = nodes.get("storyboard", {}).get("output_data", {})
        video_output = nodes.get("video_generation", {}).get("output_data", {})

        concepts_summary = ""
        for c in concepts_output.get("concepts", [])[:3]:
            concepts_summary += f"- {c.get('title', 'Untitled')}: {c.get('hook', '')}\n"

        storyboard_summary = ""
        for sb in storyboard_output.get("storyboards", [])[:2]:
            storyboard_summary += f"- Storyline: {sb.get('storyline', '')[:200]}\n"
            storyboard_summary += f"  Scenes: {len(sb.get('scenes', []))} cuts\n"

        # Get stitched/video variations
        all_variations = video_output.get("variations", [])
        full_videos = [v for v in all_variations if v.get("type") in ("stitched", "video") and v.get("preview")]
        if body.video_ids:
            full_videos = [v for v in full_videos if v.get("id") in body.video_ids]
        if not full_videos:
            raise HTTPException(status_code=400, detail="No stitched/video variations found to predict for")

        # Research benchmarks
        benchmarks = _compute_research_benchmarks(config)
        brand_bench = benchmarks.get("brand", {})
        comp_bench = benchmarks.get("competitors", {})

        benchmark_context = ""
        if brand_bench:
            benchmark_context += f"\nBRAND BENCHMARKS (from top-performing reels):\n"
            benchmark_context += f"  Followers: {brand_bench.get('followers', 'N/A'):,}\n"
            benchmark_context += f"  Avg Views: {brand_bench.get('avg_views', 0):,}\n"
            benchmark_context += f"  Avg Likes: {brand_bench.get('avg_likes', 0):,}\n"
            benchmark_context += f"  Avg Comments: {brand_bench.get('avg_comments', 0):,}\n"
            benchmark_context += f"  Engagement Rate: {brand_bench.get('engagement_rate', 0)}%\n"

        if comp_bench:
            benchmark_context += "\nCOMPETITOR BENCHMARKS:\n"
            for comp_user, kpis in comp_bench.items():
                benchmark_context += f"  {comp_user}: {kpis.get('followers', 0):,} followers, avg views {kpis.get('avg_views', 0):,}, avg likes {kpis.get('avg_likes', 0):,}, engagement {kpis.get('engagement_rate', 0)}%\n"

        # Resolve provider/model
        llm_model = body.model_name
        if llm_model.startswith("gemini") or llm_model == "gemini-pro-3":
            provider = "google"
        elif llm_model.startswith("gpt") or llm_model.startswith("o1"):
            provider = "openai"
        elif llm_model.startswith("claude"):
            provider = "anthropic"
        else:
            provider = "google"

        model_map = {
            "gemini-pro-3": "gemini-2.5-pro",
            "claude-4.5-sonnet": "claude-sonnet-4.5",
            "gpt-5.2": "gpt-4o",
        }
        mapped_model = model_map.get(llm_model, llm_model)

        from src.ai_agent import evaluate_with_openai, evaluate_with_anthropic, evaluate_with_google

        predictions = []
        for vid in full_videos:
            vid_id = vid.get("id", "unknown")
            vid_title = vid.get("title", "Untitled")
            vid_model = vid.get("model", "")
            vid_type = vid.get("type", "")

            video_desc = f"{vid_title}"
            if vid_model:
                video_desc += f" (generated with {vid_model})"
            if vid_type == "stitched":
                video_desc += " — full stitched video from all scenes"

            prompt = f"""You are an expert social media marketing analyst specializing in Instagram Reels performance prediction.

Given the brand context, content details, and real Instagram benchmark data below, predict the expected performance KPIs for this specific video if posted as an Instagram Reel.

BRAND: {brand_name}
{f"Brand Description: {brand_description}" if brand_description else ""}
{f"Campaign: {campaign_name}" if campaign_name else ""}
{f"Campaign Goal: {campaign_goal}" if campaign_goal else ""}

CONTENT CONTEXT:
Concepts:
{concepts_summary if concepts_summary else "No concepts available."}

Storyboards:
{storyboard_summary if storyboard_summary else "No storyboards available."}

VIDEO BEING EVALUATED: {video_desc}
{benchmark_context}

Based on the benchmark data and content analysis, predict the following KPIs for this video:

Return ONLY a JSON object with this exact structure:
{{"expected_views": <int>, "expected_likes": <int>, "expected_comments": <int>, "engagement_rate": <float percent>, "confidence": <float 0-1>, "reasoning": "<2-3 sentence explanation>", "strengths": ["<strength1>", "<strength2>"], "risks": ["<risk1>", "<risk2>"]}}

Guidelines:
- Base predictions on the brand's actual benchmark data
- Consider the content type, hook quality, and production value
- engagement_rate = (likes + comments) / followers * 100
- confidence should reflect how similar this content is to successful benchmarks
- Be realistic — don't inflate numbers beyond what benchmarks suggest is achievable"""

            result = None
            if provider == "openai":
                result = evaluate_with_openai(prompt, mapped_model)
            elif provider == "anthropic":
                result = evaluate_with_anthropic(prompt, mapped_model)
            elif provider == "google":
                result = evaluate_with_google(prompt, mapped_model)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

            if result:
                result["video_id"] = vid_id
                result["video_title"] = vid_title
                # Clamp values
                for k in ("expected_views", "expected_likes", "expected_comments"):
                    result[k] = max(0, int(result.get(k, 0)))
                result["engagement_rate"] = round(max(0, float(result.get("engagement_rate", 0))), 2)
                result["confidence"] = round(max(0, min(1, float(result.get("confidence", 0.5)))), 2)
                predictions.append(result)

        if not predictions:
            raise HTTPException(status_code=500, detail="LLM returned no predictions")

        # Save to predictive_modeling node
        output_data = {
            "predictions": predictions,
            "benchmarks": benchmarks,
            "model_name": llm_model,
            "run_at": datetime.utcnow().isoformat(),
        }
        pred_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "predictive_modeling",
        })
        if pred_node:
            db.content_workflow_nodes.update_one(
                {"_id": pred_node["_id"]},
                {"$set": {
                    "output_data": output_data,
                    "status": "completed",
                    "updated_at": datetime.utcnow(),
                }},
            )
        else:
            stage_index = STAGE_ORDER.index("predictive_modeling") if "predictive_modeling" in STAGE_ORDER else 9
            db.content_workflow_nodes.insert_one({
                "workflow_id": workflow_id,
                "stage_key": "predictive_modeling",
                "stage_index": stage_index,
                "stage_type": "agent",
                "status": "completed",
                "input_data": {},
                "output_data": output_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })

        _recalculate_current_stage(workflow_id)
        return {"predictions": predictions, "benchmarks": benchmarks}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Content Ranking ---


class RankRequest(BaseModel):
    simulation_weight: float = Field(default=0.4, ge=0, le=1)
    prediction_weight: float = Field(default=0.6, ge=0, le=1)


@router.post("/{workflow_id}/rank")
async def run_content_ranking(workflow_id: str, body: RankRequest, request: Request):
    """Deterministic content ranking combining simulation scores + predicted KPIs."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        nodes = {n["stage_key"]: n for n in db.content_workflow_nodes.find({"workflow_id": workflow_id})}

        # Get simulation results
        sim_output = nodes.get("simulation_testing", {}).get("output_data", {})
        sim_results = sim_output.get("results", [])

        # Also check for test-based results
        sim_tests = sim_output.get("tests", [])
        for t in sim_tests:
            if isinstance(t, dict) and t.get("results"):
                sim_results.extend(t["results"])

        # Get prediction results
        pred_output = nodes.get("predictive_modeling", {}).get("output_data", {})
        predictions = pred_output.get("predictions", [])

        if not predictions:
            raise HTTPException(status_code=400, detail="No prediction results found. Run Predictive Modeling first.")

        # Compute avg simulation score per video
        sim_scores_by_video = {}
        for r in sim_results:
            vid_id = r.get("video_id", "all")
            sim_scores_by_video.setdefault(vid_id, []).append(r.get("score", 0))

        sim_avg_by_video = {}
        for vid_id, scores in sim_scores_by_video.items():
            sim_avg_by_video[vid_id] = sum(scores) / len(scores) if scores else 0

        # Compute normalized prediction score per video (0-100 scale)
        # Based on how predicted engagement compares to benchmarks
        benchmarks = pred_output.get("benchmarks", {})
        brand_bench = benchmarks.get("brand", {})
        brand_avg_views = brand_bench.get("avg_views", 1) or 1
        brand_avg_likes = brand_bench.get("avg_likes", 1) or 1

        pred_scores = {}
        for p in predictions:
            vid_id = p.get("video_id", "unknown")
            # Score based on: confidence * (predicted performance vs benchmark ratio), clamped to 0-100
            view_ratio = min(p.get("expected_views", 0) / brand_avg_views, 2.0)  # cap at 2x benchmark
            like_ratio = min(p.get("expected_likes", 0) / brand_avg_likes, 2.0)
            confidence = p.get("confidence", 0.5)
            # Weighted average of ratios, scaled to 0-100
            raw_score = (view_ratio * 0.5 + like_ratio * 0.3 + confidence * 0.2) * 50
            pred_scores[vid_id] = round(min(100, max(0, raw_score)), 1)

        # Normalize weights
        total_weight = body.simulation_weight + body.prediction_weight
        sim_w = body.simulation_weight / total_weight if total_weight > 0 else 0.4
        pred_w = body.prediction_weight / total_weight if total_weight > 0 else 0.6

        # Build ranked list
        ranked = []
        for p in predictions:
            vid_id = p.get("video_id", "unknown")
            sim_score = sim_avg_by_video.get(vid_id, sim_avg_by_video.get("all", 50))
            pred_score = pred_scores.get(vid_id, 50)
            composite = round(sim_w * sim_score + pred_w * pred_score, 1)

            ranked.append({
                "video_id": vid_id,
                "video_title": p.get("video_title", "Untitled"),
                "composite_score": composite,
                "simulation_score": round(sim_score, 1),
                "prediction_score": pred_score,
                "expected_views": p.get("expected_views", 0),
                "expected_likes": p.get("expected_likes", 0),
                "expected_comments": p.get("expected_comments", 0),
                "engagement_rate": p.get("engagement_rate", 0),
                "confidence": p.get("confidence", 0),
                "reasoning": p.get("reasoning", ""),
            })

        ranked.sort(key=lambda x: x["composite_score"], reverse=True)

        # Add rank numbers
        for i, r in enumerate(ranked):
            r["rank"] = i + 1

        # Save to content_ranking node
        output_data = {
            "rankings": ranked,
            "weights": {"simulation": sim_w, "prediction": pred_w},
            "run_at": datetime.utcnow().isoformat(),
        }
        rank_node = db.content_workflow_nodes.find_one({
            "workflow_id": workflow_id,
            "stage_key": "content_ranking",
        })
        if rank_node:
            db.content_workflow_nodes.update_one(
                {"_id": rank_node["_id"]},
                {"$set": {
                    "output_data": output_data,
                    "status": "completed",
                    "updated_at": datetime.utcnow(),
                }},
            )
        else:
            stage_index = STAGE_ORDER.index("content_ranking") if "content_ranking" in STAGE_ORDER else 10
            db.content_workflow_nodes.insert_one({
                "workflow_id": workflow_id,
                "stage_key": "content_ranking",
                "stage_index": stage_index,
                "stage_type": "agent",
                "status": "completed",
                "input_data": {},
                "output_data": output_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })

        _recalculate_current_stage(workflow_id)
        return {"rankings": ranked, "weights": {"simulation": sim_w, "prediction": pred_w}}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Helpers ---

def _verify_workflow_access(workflow_id: str, workos_user_id: str):
    """Verify user has access to this workflow's organization. API key bypasses org check."""
    workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workos_user_id != "api_key":
        from src.role_helpers import verify_org_membership
        verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow
