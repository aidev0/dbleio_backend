#!/usr/bin/env python3
"""
Video Synthesis routes and models
Handles synthesis plan generation and video production
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
import json
import time
import tempfile
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
    print("Warning: GOOGLE_API_KEY not found")

# Create router
router = APIRouter(prefix="/api/synthesis", tags=["synthesis"])

# Pydantic Models
class SynthesisSegment(BaseModel):
    """A segment in the synthesis timeline"""
    source_video_num: int
    source_video_id: Optional[str] = None
    source_start: int
    source_end: int
    output_start: int
    output_end: int
    purpose: str
    description: str


class SynthesisRequest(BaseModel):
    """Request to generate video synthesis"""
    campaign_id: str
    description: str  # User's description of what they want


class SynthesisResponse(BaseModel):
    """Response from synthesis generation"""
    description: str  # Overall description of the synthesis
    total_duration: int  # Total duration in seconds
    timeline: List[SynthesisSegment]  # Edit timeline
    recommendations: List[str] = Field(default_factory=list)  # AI recommendations


class SynthesisVideoRequest(BaseModel):
    """Request to produce synthesis video"""
    campaign_id: str
    synthesis_plan_id: Optional[str] = None  # If not provided, uses most recent plan


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
    video_url: Optional[str] = None


# Routes

@router.post("/plans", response_model=SynthesisResponse)
async def generate_synthesis_plan(request: SynthesisRequest):
    """Generate an AI-powered video synthesis timeline based on user's description"""
    try:
        campaign_id = request.campaign_id

        # Get campaign
        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

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

USER'S REQUEST: {request.description}

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

        # Generate synthesis with Gemini 3 Pro Preview (supports multiple videos)
        model = genai.GenerativeModel('gemini-3-pro-preview')

        # Build content array with all videos
        content = []
        for v in gemini_videos:
            content.append(v["file"])
        content.append(prompt)

        print("Generating synthesis with Gemini 3 Pro Preview...")
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
            "user_description": request.description,
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


@router.get("/plans")
async def get_synthesis_plans(campaign_id: Optional[str] = None):
    """Get synthesis plans, optionally filtered by campaign"""
    try:
        query = {}
        if campaign_id:
            query["campaign_id"] = campaign_id

        plans = list(db.synthesis_plans.find(query).sort("created_at", -1))

        for plan in plans:
            plan["_id"] = str(plan["_id"])

        return plans
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/plans/{plan_id}")
async def update_synthesis_plan(plan_id: str, request: dict):
    """Update a synthesis plan"""
    try:
        # Convert plan_id to ObjectId
        from bson import ObjectId

        if not ObjectId.is_valid(plan_id):
            raise HTTPException(status_code=400, detail="Invalid plan ID")

        plan_obj_id = ObjectId(plan_id)

        # Get existing plan
        existing_plan = db.synthesis_plans.find_one({"_id": plan_obj_id})
        if not existing_plan:
            raise HTTPException(status_code=404, detail="Synthesis plan not found")

        # Update the synthesis_plan field
        update_data = {}
        if "synthesis_plan" in request:
            update_data["synthesis_plan"] = request["synthesis_plan"]

        if update_data:
            db.synthesis_plans.update_one(
                {"_id": plan_obj_id},
                {"$set": update_data}
            )

        # Return updated plan
        updated_plan = db.synthesis_plans.find_one({"_id": plan_obj_id})
        updated_plan["_id"] = str(updated_plan["_id"])

        return updated_plan
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/videos", response_model=SynthesisVideoResponse)
async def produce_synthesis_video(request: SynthesisVideoRequest):
    """Produce an actual synthesis video from a synthesis plan"""
    try:
        campaign_id = request.campaign_id

        # Get campaign
        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Get synthesis plan
        if request.synthesis_plan_id:
            synthesis_plan = db.synthesis_plans.find_one({"_id": ObjectId(request.synthesis_plan_id)})
        else:
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
            completed_at=synthesis_video.get("completed_at"),
            video_url=synthesis_video.get("video_url")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Video production error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
async def get_synthesis_videos(campaign_id: Optional[str] = None):
    """Get synthesis videos, optionally filtered by campaign"""
    try:
        query = {}
        if campaign_id:
            query["campaign_id"] = campaign_id

        videos = list(db.synthesis_videos.find(query).sort("created_at", -1))

        for video in videos:
            video["_id"] = str(video["_id"])

        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}")
async def get_synthesis_video(video_id: str):
    """Get a single synthesis video by ID"""
    try:
        video = db.synthesis_videos.find_one({"_id": ObjectId(video_id)})
        if not video:
            raise HTTPException(status_code=404, detail="Synthesis video not found")

        video["_id"] = str(video["_id"])
        return video
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
