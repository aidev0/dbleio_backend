#!/usr/bin/env python3
"""
Video routes and models
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Google Cloud Storage Configuration
GCS_BUCKET = os.getenv('GCS_BUCKET', 'video-marketing-simulation')

# Load Google Cloud credentials from environment
def get_google_credentials():
    """Load Google Cloud credentials from environment variable"""
    service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        try:
            # Parse JSON string from environment
            credentials_dict = json.loads(service_account_json)
            # Create credentials from dict
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return credentials
        except Exception as e:
            print(f"Error loading service account from env: {e}")
            return None
    return None

# Initialize GCS client with credentials
try:
    credentials = get_google_credentials()
    if credentials:
        storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
        gcs_bucket = storage_client.bucket(GCS_BUCKET)
        print(f"✓ GCS client initialized with project: {credentials.project_id}")
    else:
        # Fallback to default credentials
        storage_client = storage.Client()
        gcs_bucket = storage_client.bucket(GCS_BUCKET)
        print("✓ GCS client initialized with default credentials")
except Exception as e:
    print(f"Warning: Failed to initialize GCS client: {e}")
    storage_client = None
    gcs_bucket = None

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"✓ Gemini API configured")
else:
    print("Warning: GOOGLE_API_KEY not found in environment")

# Create router
router = APIRouter(prefix="/api/videos", tags=["videos"])

# Pydantic Models
class TimelineSegment(BaseModel):
    """A segment of the video timeline"""
    start: int  # Start time in seconds
    end: int    # End time in seconds
    description: str  # What's happening in this segment
    elements: List[str] = Field(default_factory=list)  # Visual elements shown
    purpose: str  # e.g., "hook", "product_showcase", "feature_demo", "social_proof", "call_to_action"

class VideoAnalysis(BaseModel):
    """Video understanding/analysis data"""
    summary: Optional[str] = None
    objects: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    texture: Optional[str] = None
    textures: List[str] = Field(default_factory=list)
    number_of_scene_cut: Optional[int] = None
    qualities_demonstrated: List[str] = Field(default_factory=list)
    duration_analysis: Optional[Union[int, str]] = None  # Accept both int and str for backwards compatibility
    timestamp_most_important_info_shown: List[int] = Field(default_factory=list)
    timeline: List[TimelineSegment] = Field(default_factory=list)

class VideoCreate(BaseModel):
    campaign_id: str
    title: str
    url: Optional[str] = None  # If already uploaded to S3
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    analyze_video: bool = Field(default=True)  # Whether to run AI analysis

class VideoUpdate(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None

class VideoResponse(BaseModel):
    id: str
    campaign_id: str
    title: str
    url: str
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None
    analysis: Optional[VideoAnalysis] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

# Helper functions
def get_fresh_signed_url(gs_uri: str) -> Optional[str]:
    """Generate a fresh signed URL from a GCS URI (gs://bucket/path)"""
    if not gs_uri or not gs_uri.startswith('gs://') or not gcs_bucket:
        return None
    try:
        # Extract blob path from gs://bucket/path
        path = gs_uri.replace(f'gs://{GCS_BUCKET}/', '')
        blob = gcs_bucket.blob(path)
        creds = get_google_credentials()
        signed_url = blob.generate_signed_url(
            version="v2",
            expiration=timedelta(days=7),
            method="GET",
            credentials=creds
        )
        return signed_url
    except Exception as e:
        print(f"Error generating signed URL for {gs_uri}: {e}")
        return None

def video_helper(video, include_analysis=True) -> dict:
    """Convert MongoDB video to dict"""
    # If video has a gs_uri, generate a fresh signed URL
    video_url = video["url"]
    gs_uri = video.get("gs_uri")
    if gs_uri:
        fresh_url = get_fresh_signed_url(gs_uri)
        if fresh_url:
            video_url = fresh_url

    result = {
        "id": str(video["_id"]),
        "campaign_id": video["campaign_id"],
        "title": video["title"],
        "url": video_url,
        "thumbnail_url": video.get("thumbnail_url"),
        "duration": video.get("duration"),
        "created_at": video.get("created_at", datetime.utcnow()),
        "updated_at": video.get("updated_at"),
    }

    # Optionally include analysis (get most recent analysis)
    if include_analysis:
        # Get the most recent analysis by sorting by created_at descending
        understanding = db.video_understandings.find_one(
            {"video_id": video["_id"]},
            sort=[("created_at", -1)]  # -1 for descending order
        )
        if understanding:
            result["analysis"] = {
                "summary": understanding.get("summary"),
                "objects": understanding.get("objects", []),
                "colors": understanding.get("colors", []),
                "texture": understanding.get("texture"),
                "textures": understanding.get("textures", []),
                "number_of_scene_cut": understanding.get("number_of_scene_cut"),
                "qualities_demonstrated": understanding.get("qualities_demonstrated", []),
                "duration_analysis": understanding.get("duration"),
                "timestamp_most_important_info_shown": understanding.get("timestamp_most_important_info_shown", []),
                "timeline": understanding.get("timeline", []),
                "analysis_version": understanding.get("analysis_version"),  # Include which version this is
            }

    return result

def analyze_video_with_google_ai(video_url: str) -> Dict[str, Any]:
    """
    Analyze video using Gemini 2.5 Pro for comprehensive marketing insights
    Returns structured analysis matching the video understanding schema
    """
    try:
        if not GOOGLE_API_KEY:
            raise Exception("GOOGLE_API_KEY not configured")

        print(f"Processing video analysis with Gemini 3 Pro Preview for {video_url}...")

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-3-pro-preview')

        # Handle different video URL formats
        video_input = None
        if video_url.startswith('gs://'):
            # For GCS URIs, we need to download temporarily or use File API
            print("Downloading video from GCS...")
            import tempfile
            import requests
            from google.cloud import storage as gcs_storage

            # Parse gs:// URI
            bucket_name = video_url.split('/')[2]
            blob_path = '/'.join(video_url.split('/')[3:])

            # Download to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            credentials = get_google_credentials()
            if credentials:
                gcs_client = gcs_storage.Client(credentials=credentials)
            else:
                gcs_client = gcs_storage.Client()

            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(temp_file.name)
            temp_file.close()

            print(f"Uploading video to Gemini from {temp_file.name}...")
            video_input = genai.upload_file(path=temp_file.name)

            # Wait for video processing
            print("Processing video...")
            import time
            while video_input.state.name == "PROCESSING":
                time.sleep(2)
                video_input = genai.get_file(video_input.name)

            if video_input.state.name == "FAILED":
                raise Exception("Video processing failed")

            # Clean up temp file
            import os as os_module
            try:
                os_module.unlink(temp_file.name)
            except:
                pass

        elif video_url.startswith('http://') or video_url.startswith('https://'):
            # For HTTP URLs (signed URLs), download and upload
            print("Downloading video from URL...")
            import tempfile
            import requests

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()

            print(f"Uploading video to Gemini from {temp_file.name}...")
            video_input = genai.upload_file(path=temp_file.name)

            # Wait for video processing
            print("Processing video...")
            import time
            while video_input.state.name == "PROCESSING":
                time.sleep(2)
                video_input = genai.get_file(video_input.name)

            if video_input.state.name == "FAILED":
                raise Exception("Video processing failed")

            # Clean up temp file
            import os as os_module
            try:
                os_module.unlink(temp_file.name)
            except:
                pass
        else:
            # Local file path
            print(f"Uploading local video to Gemini...")
            video_input = genai.upload_file(path=video_url)

            # Wait for video processing
            print("Processing video...")
            import time
            while video_input.state.name == "PROCESSING":
                time.sleep(2)
                video_input = genai.get_file(video_input.name)

            if video_input.state.name == "FAILED":
                raise Exception("Video processing failed")

        # Comprehensive marketing analysis prompt
        prompt = """Analyze this marketing video comprehensively for a marketing simulation platform. Provide detailed insights about:

1. **Summary**: A 2-3 sentence overview of the video's content, messaging, and marketing approach
2. **Objects**: List all visible products, items, people, locations, and significant elements shown
3. **Colors**: Identify the dominant color palette and visual themes (e.g., "vibrant red", "soft pastels", "monochrome")
4. **Textures**: Describe materials, surfaces, and visual textures (e.g., "glossy", "matte", "organic", "metallic")
5. **Scene Cuts**: Count the number of COHESIVE SCENE CUTS where the CONTEXT or SETTING changes significantly (not just camera angles). A cohesive scene cut is when the video transitions to a new context, location, subject, or narrative moment. Count carefully - only count transitions where the context meaningfully shifts.
6. **Qualities Demonstrated**: List product features, benefits, or brand values shown (e.g., "durability", "elegance", "innovation")
7. **Duration**: The total length of the video in seconds (e.g., 30, 60, 90)
8. **Key Timestamps**: Identify timestamps (in seconds) where the most important information or compelling moments appear
9. **Timeline**: Break down the video into segments based on COHESIVE SCENE CUTS. Each segment should represent a distinct context or narrative moment. For each segment provide:
   - start/end timestamps (in seconds)
   - description of what's happening in this scene
   - visual elements shown
   - marketing purpose using ONE of these categories:
     * hook - Opening that grabs attention
     * product_showcase - Displaying the product/brand
     * feature_showcase - Demonstrating specific features
     * benefit_highlight - Showing benefits/results
     * problem_setup - Establishing customer pain point
     * solution_reveal - Showing how product solves problem
     * social_proof - Testimonials, reviews, or credibility
     * call_to_action - Encouraging viewer action
     * brand_display - Logo, branding elements
     * demonstration - Step-by-step how-to
     * comparison - Before/after or vs competitors
     * emotional_appeal - Evoking emotions or lifestyle

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{
  "summary": "detailed 2-3 sentence description",
  "objects": ["list", "of", "objects", "and", "elements"],
  "colors": ["color1", "color2", "color3"],
  "textures": ["texture1", "texture2"],
  "number_of_scene_cut": 10,
  "qualities_demonstrated": ["quality1", "quality2"],
  "duration": 60,
  "timestamp_most_important_info_shown": [5, 12, 28],
  "timeline": [
    {
      "start": 0,
      "end": 5,
      "description": "Opening hook showing customer problem or pain point",
      "elements": ["person", "frustrated expression", "messy environment"],
      "purpose": "hook"
    },
    {
      "start": 5,
      "end": 12,
      "description": "Product introduction with prominent branding and logo display",
      "elements": ["product", "logo", "brand colors", "text overlay"],
      "purpose": "product_showcase"
    },
    {
      "start": 12,
      "end": 20,
      "description": "Demonstration of key product feature in action",
      "elements": ["product in use", "hands", "feature highlight"],
      "purpose": "feature_showcase"
    },
    {
      "start": 20,
      "end": 28,
      "description": "Before/after comparison showing transformation",
      "elements": ["split screen", "messy vs clean", "satisfied customer"],
      "purpose": "comparison"
    },
    {
      "start": 28,
      "end": 30,
      "description": "Call to action with website and discount code",
      "elements": ["text overlay", "website URL", "promo code"],
      "purpose": "call_to_action"
    }
  ]
}

IMPORTANT: Base timeline segments on COHESIVE SCENE CUTS. Each segment should represent a distinct scene where the context changes. The number of timeline segments should closely match the number_of_scene_cut count."""

        # Generate analysis
        response = model.generate_content([video_input, prompt])

        # Parse JSON response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        analysis = json.loads(response_text)

        print(f"✓ Video analysis completed successfully")
        return analysis

    except Exception as e:
        print(f"Error analyzing video with Gemini: {e}")
        import traceback
        traceback.print_exc()

        # Return basic empty analysis if AI fails
        return {
            "summary": f"Video analysis pending (Error: {str(e)})",
            "objects": [],
            "colors": [],
            "textures": [],
            "number_of_scene_cut": 0,
            "qualities_demonstrated": [],
            "duration": 0,
            "timestamp_most_important_info_shown": [],
            "timeline": [],
        }

def upload_to_gcs(file: UploadFile, campaign_id: str) -> Dict[str, str]:
    """
    Upload file to Google Cloud Storage and return URLs
    Returns: {"url": str, "gs_uri": str, "thumbnail_url": str}
    """
    if not gcs_bucket:
        raise HTTPException(
            status_code=500,
            detail="GCS client not configured. Please set GOOGLE_APPLICATION_CREDENTIALS."
        )

    try:
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        blob_name = f"campaigns/{campaign_id}/videos/{timestamp}_{file.filename}"

        # Create blob and upload
        blob = gcs_bucket.blob(blob_name)

        # Reset file position to beginning
        file.file.seek(0)

        # Upload to GCS
        blob.upload_from_file(
            file.file,
            content_type=file.content_type
        )

        # Generate URLs
        # GCS URI for Video Intelligence API (gs://bucket/path)
        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

        # Generate a signed URL (valid for 7 days) for viewing
        creds = get_google_credentials()
        video_url = blob.generate_signed_url(
            version="v2",
            expiration=timedelta(days=7),
            method="GET",
            credentials=creds
        )

        # For thumbnail, we'll use a placeholder or generate one later
        thumbnail_url = None

        return {
            "url": video_url,
            "gs_uri": gs_uri,
            "thumbnail_url": thumbnail_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to GCS: {str(e)}")

# Routes
@router.get("", response_model=List[VideoResponse])
async def get_videos(campaign_id: Optional[str] = None):
    """Get all videos, optionally filtered by campaign"""
    try:
        query = {}
        if campaign_id:
            query["campaign_id"] = campaign_id

        videos = db.videos.find(query)
        return [video_helper(video) for video in videos]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str):
    """Get a single video by ID"""
    try:
        video = db.videos.find_one({"_id": ObjectId(video_id)})
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video_helper(video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def create_video(video: VideoCreate):
    """Create a new video (without file upload - URL already provided)"""
    try:
        video_dict = video.model_dump(exclude={"analyze_video"})
        video_dict["created_at"] = datetime.utcnow()
        video_dict["updated_at"] = datetime.utcnow()

        # Insert video
        result = db.videos.insert_one(video_dict)
        video_id = result.inserted_id

        # Run AI analysis if requested and URL is provided
        if video.analyze_video and video.url:
            try:
                analysis = analyze_video_with_google_ai(video.url)

                # Store video understanding
                understanding = {
                    "video_id": video_id,
                    "campaign_id": video.campaign_id,
                    "created_at": datetime.utcnow(),
                    **analysis
                }
                db.video_understandings.insert_one(understanding)
            except Exception as e:
                print(f"Warning: Failed to analyze video: {e}")
                # Continue even if analysis fails

        new_video = db.videos.find_one({"_id": video_id})
        return video_helper(new_video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def analyze_video_background(video_id: ObjectId, campaign_id: str, gs_uri: str, task_id: str):
    """Background task to analyze video with AI"""
    from src.task_manager import task_manager, TaskStatus

    try:
        print(f"Starting background video analysis for video {video_id}")
        task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10, message="Uploading video to AI service...")

        analysis = analyze_video_with_google_ai(gs_uri)

        task_manager.update_task(task_id, progress=90, message="Saving analysis results...")

        # Store video understanding
        understanding = {
            "video_id": video_id,
            "campaign_id": campaign_id,
            "created_at": datetime.utcnow(),
            **analysis
        }
        db.video_understandings.insert_one(understanding)

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Video analysis completed",
            result={"video_id": str(video_id)}
        )
        print(f"✓ Background video analysis completed for video {video_id}")
    except Exception as e:
        error_msg = f"Error in background video analysis: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, message=error_msg, error=str(e))

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    campaign_id: str = Form(...),
    title: str = Form(...),
    analyze_video: bool = Form(True)
):
    """Upload a video file to Google Cloud Storage and create video record"""
    from src.task_manager import task_manager
    import uuid

    try:
        # Validate file type
        allowed_types = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/mpeg']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )

        # Upload to GCS
        urls = upload_to_gcs(file, campaign_id)

        # Create video record
        video_dict = {
            "campaign_id": campaign_id,
            "title": title,
            "url": urls["url"],
            "gs_uri": urls.get("gs_uri"),  # Store GCS URI for Video Intelligence API
            "thumbnail_url": urls.get("thumbnail_url"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = db.videos.insert_one(video_dict)
        video_id = result.inserted_id

        # Create and schedule AI analysis task if requested
        task_id = None
        if analyze_video:
            gs_uri = urls.get("gs_uri")
            if gs_uri:
                task_id = f"video_analysis_{video_id}_{uuid.uuid4().hex[:8]}"
                task_manager.create_task(
                    task_id,
                    "video_analysis",
                    metadata={"video_id": str(video_id), "campaign_id": campaign_id}
                )
                background_tasks.add_task(analyze_video_background, video_id, campaign_id, gs_uri, task_id)
                print(f"Scheduled background analysis for video {video_id} with task {task_id}")

        new_video = db.videos.find_one({"_id": video_id})
        response_data = video_helper(new_video)

        # Add task_id to response if analysis is scheduled
        if task_id:
            response_data["task_id"] = task_id

        return response_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{video_id}", response_model=VideoResponse)
async def update_video(video_id: str, video_update: VideoUpdate):
    """Update an existing video"""
    try:
        # Check if video exists
        existing_video = db.videos.find_one({"_id": ObjectId(video_id)})
        if not existing_video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Update fields
        update_dict = video_update.model_dump(exclude_unset=True)
        update_dict["updated_at"] = datetime.utcnow()

        db.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": update_dict}
        )

        updated_video = db.videos.find_one({"_id": ObjectId(video_id)})
        return video_helper(updated_video)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(video_id: str):
    """Delete a video and its analysis"""
    try:
        # Delete video understanding first
        db.video_understandings.delete_many({"video_id": ObjectId(video_id)})

        # Delete video
        result = db.videos.delete_one({"_id": ObjectId(video_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{video_id}/analyze", response_model=VideoResponse)
async def analyze_existing_video(video_id: str):
    """Trigger AI analysis for an existing video"""
    try:
        # Get video
        video = db.videos.find_one({"_id": ObjectId(video_id)})
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Prefer GCS URI for Video Intelligence API, fallback to public URL
        video_uri = video.get("gs_uri") or video.get("url")
        if not video_uri:
            raise HTTPException(status_code=400, detail="Video has no URI to analyze")

        # Run AI analysis
        analysis = analyze_video_with_google_ai(video_uri)

        # Store new video understanding (append to history, don't delete old ones)
        understanding = {
            "video_id": ObjectId(video_id),
            "campaign_id": video["campaign_id"],
            "created_at": datetime.utcnow(),
            "analysis_version": datetime.utcnow().isoformat(),  # Track which analysis this is
            **analysis
        }
        db.video_understandings.insert_one(understanding)

        # Return updated video with analysis
        updated_video = db.videos.find_one({"_id": ObjectId(video_id)})
        return video_helper(updated_video)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
