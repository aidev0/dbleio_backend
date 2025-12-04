#!/usr/bin/env python3
"""
Campaign routes and models
"""

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from auth import get_workos_user_id, require_user_id
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
import json
import tempfile
import time
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"✓ Gemini API configured for synthesis")
else:
    print("Warning: GOOGLE_API_KEY not found in environment")

# Create router
router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])

# Pydantic Models
class Advertiser(BaseModel):
    business_name: Optional[str] = None
    website_url: Optional[str] = None

class PerformanceObjective(BaseModel):
    value: Optional[float] = None
    kpi: Optional[str] = None  # ROAS, CPA, CPL, CPC, CTR, etc.

class AudienceControl(BaseModel):
    location: Optional[List[str]] = Field(default_factory=list)  # Countries
    zip_codes: Optional[List[str]] = Field(default_factory=list)
    in_market_interests: Optional[List[str]] = Field(default_factory=list)

class Strategy(BaseModel):
    name: Optional[str] = None
    budget_amount: Optional[float] = None
    budget_type: Optional[str] = None  # daily, weekly, monthly
    audience_control: Optional[AudienceControl] = Field(default_factory=AudienceControl)

class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    platform: Optional[str] = None  # vibe.co, facebook, instagram, linkedin, tiktok, x
    advertiser: Optional[Advertiser] = Field(default_factory=Advertiser)
    campaign_goal: Optional[str] = None  # awareness, traffic, leads, sales, retargeting, app_promotion
    performance_objective: Optional[PerformanceObjective] = Field(default_factory=PerformanceObjective)
    strategies: Optional[List[Strategy]] = Field(default_factory=list)
    user_id: Optional[str] = None  # MongoDB _id of the user who owns this campaign
    workos_user_id: Optional[str] = None  # WorkOS user ID of the user who owns this campaign
    shared_ids: Optional[List[str]] = Field(default_factory=list)  # List of workos_user_ids this campaign is shared with

class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    platform: Optional[str] = None
    advertiser: Optional[Advertiser] = None
    campaign_goal: Optional[str] = None
    performance_objective: Optional[PerformanceObjective] = None
    strategies: Optional[List[Strategy]] = None
    user_id: Optional[str] = None
    workos_user_id: Optional[str] = None
    shared_ids: Optional[List[str]] = None

class CampaignResponse(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    underscore_id: str = Field(alias="_id", serialization_alias="_id")
    name: str
    description: Optional[str] = None
    platform: Optional[str] = None
    advertiser: Advertiser = Field(default_factory=Advertiser)
    campaign_goal: Optional[str] = None
    performance_objective: PerformanceObjective = Field(default_factory=PerformanceObjective)
    strategies: List[Strategy] = Field(default_factory=list)
    user_id: Optional[str] = None
    workos_user_id: Optional[str] = None
    shared_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

# Helper functions
def verify_campaign_access(campaign_id: str, workos_user_id: str, require_owner: bool = False):
    """
    Verify user has access to a campaign.
    Returns the campaign if access is granted, raises HTTPException otherwise.

    Args:
        campaign_id: The campaign ID to check
        workos_user_id: The authenticated user's WorkOS ID
        require_owner: If True, only the owner can access (not shared users)
    """
    campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_owner = campaign.get("workos_user_id")
    shared_ids = campaign.get("shared_ids", [])

    if require_owner:
        if campaign_owner != workos_user_id:
            raise HTTPException(status_code=403, detail="Only the campaign owner can perform this action")
    else:
        if campaign_owner != workos_user_id and workos_user_id not in shared_ids:
            raise HTTPException(status_code=403, detail="Access denied to this campaign")

    return campaign


def campaign_helper(campaign) -> dict:
    """Convert MongoDB campaign to dict"""
    return {
        "id": str(campaign["_id"]),
        "_id": str(campaign["_id"]),
        "name": campaign["name"],
        "description": campaign.get("description"),
        "platform": campaign.get("platform"),
        "advertiser": campaign.get("advertiser", {}),
        "campaign_goal": campaign.get("campaign_goal"),
        "performance_objective": campaign.get("performance_objective", {}),
        "strategies": campaign.get("strategies", []),
        "user_id": campaign.get("user_id"),
        "workos_user_id": campaign.get("workos_user_id"),
        "shared_ids": campaign.get("shared_ids", []),
        "created_at": campaign.get("created_at", datetime.utcnow()),
        "updated_at": campaign.get("updated_at", datetime.utcnow()),
    }

# Routes
@router.get("", response_model=List[CampaignResponse])
async def get_campaigns(request: Request):
    """Get all campaigns for the authenticated user (including campaigns shared with the user)"""
    try:
        # Get user ID from JWT - required for this endpoint
        workos_user_id = require_user_id(request)

        # Build query to include campaigns owned by user OR shared with user
        query = {
            "$or": [
                {"workos_user_id": workos_user_id},
                {"shared_ids": workos_user_id}
            ]
        }

        campaigns = db.campaigns.find(query)
        return [campaign_helper(campaign) for campaign in campaigns]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str, request: Request):
    """Get a single campaign by ID (must be owner or shared with user)"""
    try:
        # Get user ID from JWT
        workos_user_id = require_user_id(request)

        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Check if user has access (owner or shared)
        campaign_owner = campaign.get("workos_user_id")
        shared_ids = campaign.get("shared_ids", [])

        if campaign_owner != workos_user_id and workos_user_id not in shared_ids:
            raise HTTPException(status_code=403, detail="Access denied to this campaign")

        return campaign_helper(campaign)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED)
async def create_campaign(campaign: CampaignCreate, request: Request):
    """Create a new campaign (owned by authenticated user)"""
    try:
        # Get user ID from JWT - this is the campaign owner
        workos_user_id = require_user_id(request)

        # Use mode='json' and exclude_none=False to ensure all fields are included
        campaign_dict = campaign.model_dump(mode='json', exclude_none=False)

        # Override workos_user_id with authenticated user - don't trust client-provided value
        campaign_dict['workos_user_id'] = workos_user_id

        # Ensure description field is always present (even if None or empty string)
        if 'description' not in campaign_dict:
            campaign_dict['description'] = None

        campaign_dict["created_at"] = datetime.utcnow()
        campaign_dict["updated_at"] = datetime.utcnow()

        result = db.campaigns.insert_one(campaign_dict)
        new_campaign = db.campaigns.find_one({"_id": result.inserted_id})

        return campaign_helper(new_campaign)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(campaign_id: str, campaign_update: CampaignUpdate, request: Request):
    """Update an existing campaign (must be owner)"""
    try:
        # Get user ID from JWT
        workos_user_id = require_user_id(request)

        # Check if campaign exists
        existing_campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not existing_campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Only owner can update (not shared users)
        if existing_campaign.get("workos_user_id") != workos_user_id:
            raise HTTPException(status_code=403, detail="Only the campaign owner can update it")

        # Update fields - include all provided fields, even if None
        # Use exclude_unset=True to only get fields that were explicitly set in the request
        update_dict = campaign_update.model_dump(exclude_unset=True)

        # Don't allow changing ownership via update
        if 'workos_user_id' in update_dict:
            del update_dict['workos_user_id']

        update_dict["updated_at"] = datetime.utcnow()

        db.campaigns.update_one(
            {"_id": ObjectId(campaign_id)},
            {"$set": update_dict}
        )

        updated_campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        return campaign_helper(updated_campaign)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{campaign_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_campaign(campaign_id: str, request: Request):
    """Delete a campaign (must be owner)"""
    try:
        # Get user ID from JWT
        workos_user_id = require_user_id(request)

        # Check if campaign exists and user is owner
        existing_campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not existing_campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Only owner can delete
        if existing_campaign.get("workos_user_id") != workos_user_id:
            raise HTTPException(status_code=403, detail="Only the campaign owner can delete it")

        db.campaigns.delete_one({"_id": ObjectId(campaign_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Synthesis Models
class SynthesisSegment(BaseModel):
    """A segment in the synthesis timeline"""
    source_video_num: int  # Which video number (1, 2, 3...)
    source_video_id: str  # Video ID from database
    source_start: int  # Start time in source video (seconds)
    source_end: int  # End time in source video (seconds)
    output_start: int  # Start time in output video (seconds)
    output_end: int  # End time in output video (seconds)
    purpose: str  # e.g., "hook", "product_showcase", etc.
    description: str  # What's in this segment


class SynthesisRequest(BaseModel):
    """Request to generate video synthesis"""
    description: str  # User's description of what they want


class SynthesisResponse(BaseModel):
    """Response from synthesis generation"""
    description: str  # Overall description of the synthesis
    total_duration: int  # Total duration in seconds
    timeline: List[SynthesisSegment]  # Edit timeline
    recommendations: List[str] = Field(default_factory=list)  # AI recommendations


@router.post("/{campaign_id}/generate-synthesis", response_model=SynthesisResponse)
async def generate_synthesis(campaign_id: str, synthesis_request: SynthesisRequest, request: Request):
    """Generate an AI-powered video synthesis timeline based on user's description"""
    try:
        # Verify user has access to this campaign
        workos_user_id = require_user_id(request)
        campaign = verify_campaign_access(campaign_id, workos_user_id)

        # Get all videos for this campaign
        videos = list(db.videos.find({"campaign_id": campaign_id}))
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for this campaign")

        # Get video analyses
        videos_with_analysis = []
        for idx, video in enumerate(videos, 1):
            analysis = db.video_understandings.find_one(
                {"video_id": video["_id"]},
                sort=[("created_at", -1)]
            )
            if analysis:
                videos_with_analysis.append({
                    "video_num": idx,
                    "video_id": str(video["_id"]),
                    "title": video["title"],
                    "url": video["url"],
                    "duration": video.get("duration"),
                    "analysis": {
                        "summary": analysis.get("summary"),
                        "timeline": analysis.get("timeline", []),
                        "qualities_demonstrated": analysis.get("qualities_demonstrated", []),
                        "objects": analysis.get("objects", []),
                    }
                })

        if not videos_with_analysis:
            raise HTTPException(status_code=404, detail="No analyzed videos found. Please analyze videos first.")

        # Upload videos to Gemini for multimodal analysis
        print(f"Uploading {len(videos_with_analysis)} videos to Gemini...")
        gemini_videos = []
        for vid_data in videos_with_analysis:
            try:
                video_url = vid_data["url"]

                # Download and upload video to Gemini
                if video_url.startswith('gs://'):
                    # GCS URL - download first
                    print(f"Downloading video {vid_data['video_num']} from GCS...")
                    import subprocess
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    subprocess.run(['gsutil', 'cp', video_url, temp_file.name], check=True)
                    temp_file.close()

                    print(f"Uploading video {vid_data['video_num']} to Gemini...")
                    video_input = genai.upload_file(path=temp_file.name)

                    # Wait for processing
                    while video_input.state.name == "PROCESSING":
                        time.sleep(2)
                        video_input = genai.get_file(video_input.name)

                    if video_input.state.name == "FAILED":
                        print(f"Warning: Video {vid_data['video_num']} processing failed, skipping")
                        continue

                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass

                elif video_url.startswith('http://') or video_url.startswith('https://'):
                    # HTTP URL - download and upload
                    print(f"Downloading video {vid_data['video_num']} from URL...")
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    response = requests.get(video_url, stream=True)
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file.close()

                    print(f"Uploading video {vid_data['video_num']} to Gemini...")
                    video_input = genai.upload_file(path=temp_file.name)

                    # Wait for processing
                    while video_input.state.name == "PROCESSING":
                        time.sleep(2)
                        video_input = genai.get_file(video_input.name)

                    if video_input.state.name == "FAILED":
                        print(f"Warning: Video {vid_data['video_num']} processing failed, skipping")
                        continue

                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                else:
                    # Local file
                    print(f"Uploading local video {vid_data['video_num']} to Gemini...")
                    video_input = genai.upload_file(path=video_url)

                    # Wait for processing
                    while video_input.state.name == "PROCESSING":
                        time.sleep(2)
                        video_input = genai.get_file(video_input.name)

                    if video_input.state.name == "FAILED":
                        print(f"Warning: Video {vid_data['video_num']} processing failed, skipping")
                        continue

                gemini_videos.append({
                    "video_num": vid_data["video_num"],
                    "video_id": vid_data["video_id"],
                    "file": video_input,
                    "analysis": vid_data["analysis"]
                })

            except Exception as e:
                print(f"Error uploading video {vid_data['video_num']}: {e}")
                continue

        if not gemini_videos:
            raise HTTPException(status_code=500, detail="Failed to upload videos to Gemini")

        # Build the synthesis prompt
        video_analyses_text = "\n\n".join([
            f"""Video {v['video_num']}:
Summary: {v['analysis']['summary']}
Qualities: {', '.join(v['analysis']['qualities_demonstrated'])}
Timeline segments: {json.dumps(v['analysis']['timeline'], indent=2)}"""
            for v in gemini_videos
        ])

        prompt = f"""You are an expert video editor and marketing strategist. The user wants to create a synthesized video by combining segments from multiple marketing videos.

USER'S REQUEST: {synthesis_request.description}

You have access to {len(gemini_videos)} videos. Here are the analyses of each video:

{video_analyses_text}

Based on the user's request and the video content, create an optimal edit timeline that:
1. Selects the best segments from each video that match the user's intent
2. Orders them in a compelling narrative sequence
3. Creates smooth transitions between segments
4. Aims for professional pacing and flow
5. Considers the marketing purpose of each segment

Also provide:
- Recommendations for why these segments were chosen
- Suggestions for improvements or alternatives

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{{
  "description": "A clear description of what this synthesis achieves",
  "total_duration": 30,
  "recommendations": [
    "Why this combination works",
    "Alternative approaches to consider",
    "Tips for best results"
  ],
  "timeline": [
    {{
      "source_video_num": 1,
      "source_start": 0,
      "source_end": 5,
      "output_start": 0,
      "output_end": 5,
      "purpose": "hook",
      "description": "Opening segment that grabs attention"
    }},
    {{
      "source_video_num": 2,
      "source_start": 10,
      "source_end": 20,
      "output_start": 5,
      "output_end": 15,
      "purpose": "product_showcase",
      "description": "Main product demonstration"
    }}
  ]
}}

IMPORTANT:
- Base your decisions on the actual video content you can see
- Select segments that flow well together
- Consider the user's specific request when choosing segments
- Ensure output timeline is continuous (each segment's output_end = next segment's output_start)
- Purpose should be one of: hook, product_showcase, feature_showcase, benefit_highlight, problem_setup, solution_reveal, social_proof, call_to_action, brand_display, demonstration, comparison, emotional_appeal"""

        # Generate synthesis with Gemini 2.0 Flash (supports multiple videos)
        model = genai.GenerativeModel('gemini-2.5-pro')

        # Build content array with all videos
        content = []
        for v in gemini_videos:
            content.append(v["file"])
        content.append(prompt)

        print("Generating synthesis with Gemini 2.0 Flash...")
        response = model.generate_content(content)

        # Parse JSON response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        synthesis_data = json.loads(response_text)

        # Add video IDs to timeline segments
        for segment in synthesis_data["timeline"]:
            video_num = segment["source_video_num"]
            matching_video = next((v for v in gemini_videos if v["video_num"] == video_num), None)
            if matching_video:
                segment["source_video_id"] = matching_video["video_id"]
            else:
                segment["source_video_id"] = ""

        # Save synthesis to database
        synthesis_record = {
            "campaign_id": campaign_id,
            "user_description": synthesis_request.description,
            "synthesis_plan": synthesis_data,
            "videos_count": len(gemini_videos),
            "created_at": datetime.utcnow(),
            "generated_by": "gemini-2.5-pro"
        }
        db.synthesis_plans.insert_one(synthesis_record)

        return SynthesisResponse(**synthesis_data)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response_text}")
        raise HTTPException(status_code=500, detail=f"Failed to parse synthesis response: {str(e)}")
    except Exception as e:
        print(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SynthesisVideoResponse(BaseModel):
    """Response from video production"""
    id: str
    campaign_id: str
    title: str
    description: str
    status: str  # "pending", "processing", "completed", "failed"
    synthesis_plan: dict
    created_at: datetime
    completed_at: Optional[datetime] = None


@router.get("/{campaign_id}/synthesis-plans")
async def get_synthesis_plans(campaign_id: str, request: Request):
    """Get all synthesis plans for a campaign"""
    try:
        # Verify user has access to this campaign
        workos_user_id = require_user_id(request)
        verify_campaign_access(campaign_id, workos_user_id)

        plans = list(db.synthesis_plans.find(
            {"campaign_id": campaign_id}
        ).sort("created_at", -1))

        for plan in plans:
            plan["_id"] = str(plan["_id"])

        return plans
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/synthesis-videos")
async def get_synthesis_videos(campaign_id: str, request: Request):
    """Get all synthesis videos for a campaign"""
    try:
        # Verify user has access to this campaign
        workos_user_id = require_user_id(request)
        verify_campaign_access(campaign_id, workos_user_id)

        videos = list(db.synthesis_videos.find(
            {"campaign_id": campaign_id}
        ).sort("created_at", -1))

        for video in videos:
            video["_id"] = str(video["_id"])

        return videos
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{campaign_id}/produce-synthesis-video", response_model=SynthesisVideoResponse)
async def produce_synthesis_video(campaign_id: str, request: Request):
    """Produce an actual synthesis video from the most recent synthesis plan"""
    try:
        # Verify user has access to this campaign
        workos_user_id = require_user_id(request)
        campaign = verify_campaign_access(campaign_id, workos_user_id)

        # Get most recent synthesis plan for this campaign
        synthesis_plan = db.synthesis_plans.find_one(
            {"campaign_id": campaign_id},
            sort=[("created_at", -1)]
        )

        if not synthesis_plan:
            raise HTTPException(status_code=404, detail="No synthesis plan found. Generate a synthesis plan first.")

        # Get all source videos referenced in the timeline
        timeline = synthesis_plan["synthesis_plan"]["timeline"]
        video_ids = set()
        for segment in timeline:
            if "source_video_id" in segment and segment["source_video_id"]:
                try:
                    video_ids.add(ObjectId(segment["source_video_id"]))
                except:
                    pass

        # Verify all videos exist
        videos = list(db.videos.find({"_id": {"$in": list(video_ids)}}))
        if len(videos) != len(video_ids):
            raise HTTPException(status_code=404, detail="Some source videos not found")

        # Create synthesis video record
        synthesis_video = {
            "campaign_id": campaign_id,
            "title": f"AI-Synthesized Video - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "description": synthesis_plan["synthesis_plan"]["description"],
            "status": "pending",  # In production: "pending" -> "processing" -> "completed"
            "synthesis_plan": synthesis_plan["synthesis_plan"],
            "synthesis_plan_id": str(synthesis_plan["_id"]),
            "source_videos": [str(vid) for vid in video_ids],
            "total_duration": synthesis_plan["synthesis_plan"]["total_duration"],
            "timeline": timeline,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        result = db.synthesis_videos.insert_one(synthesis_video)
        synthesis_video["_id"] = result.inserted_id

        # Trigger async video rendering
        import threading
        from synthesis_video_renderer import render_synthesis_video

        def render_in_background(video_id):
            """Background thread to render video"""
            try:
                print(f"Starting background rendering for synthesis video {video_id}")
                render_synthesis_video(str(video_id))
            except Exception as e:
                print(f"Background rendering error: {e}")
                import traceback
                traceback.print_exc()

        # Start rendering in background thread
        render_thread = threading.Thread(
            target=render_in_background,
            args=(result.inserted_id,),
            daemon=True
        )
        render_thread.start()

        print(f"✓ Synthesis video {result.inserted_id} created, rendering started in background")

        return SynthesisVideoResponse(
            id=str(synthesis_video["_id"]),
            campaign_id=synthesis_video["campaign_id"],
            title=synthesis_video["title"],
            description=synthesis_video["description"],
            status=synthesis_video["status"],
            synthesis_plan=synthesis_video["synthesis_plan"],
            created_at=synthesis_video["created_at"],
            completed_at=synthesis_video.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Video production error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
