#!/usr/bin/env python3
"""
Insights routes - Interactive chat-based insights using Gemini 2.5
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
import os
import json
import uuid
import tempfile
import time
from collections import defaultdict
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

router = APIRouter(prefix="/api", tags=["insights"])

# Video cache with TTL (Time To Live)
# Structure: {session_id: {video_id: {"file": genai.File, "expires_at": datetime}}}
video_cache: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
CACHE_TTL_HOURS = 2  # Videos expire after 2 hours


class InsightQuestion(BaseModel):
    """Predefined insight question"""
    id: str
    category: str
    question: str
    description: str
    requires_videos: bool = False


class InsightChatMessage(BaseModel):
    """Chat message for insights"""
    question: str
    campaign_id: Optional[str] = None
    session_id: str
    video_ids: List[str] = []
    context: Optional[Dict[str, Any]] = None


class InsightChatResponse(BaseModel):
    """Response from insights chat"""
    answer: str
    session_id: str
    cached_videos: List[str] = []
    generated_at: datetime


class GenerateInsightsRequest(BaseModel):
    """Request to generate insights"""
    question: Optional[str] = None
    evaluations: Optional[List[Dict[str, Any]]] = None
    simulations: Optional[List[Dict[str, Any]]] = None


# Cache management functions
def clean_expired_cache():
    """Remove expired videos from cache"""
    now = datetime.utcnow()
    for session_id in list(video_cache.keys()):
        for video_id in list(video_cache[session_id].keys()):
            if video_cache[session_id][video_id]["expires_at"] < now:
                del video_cache[session_id][video_id]
        # Remove empty sessions
        if not video_cache[session_id]:
            del video_cache[session_id]


def clear_session_cache(session_id: str):
    """Clear all cached videos for a session"""
    if session_id in video_cache:
        del video_cache[session_id]


def cache_video(session_id: str, video_id: str, video_file: Any):
    """Cache a video file for a session"""
    clean_expired_cache()  # Clean up first
    video_cache[session_id][video_id] = {
        "file": video_file,
        "expires_at": datetime.utcnow() + timedelta(hours=CACHE_TTL_HOURS)
    }


def get_cached_video(session_id: str, video_id: str) -> Optional[Any]:
    """Get a cached video file"""
    clean_expired_cache()
    if session_id in video_cache and video_id in video_cache[session_id]:
        cache_entry = video_cache[session_id][video_id]
        if cache_entry["expires_at"] > datetime.utcnow():
            return cache_entry["file"]
    return None


@router.get("/insights/questions", response_model=List[InsightQuestion])
async def get_insight_questions():
    """
    Get list of predefined insight questions users can ask
    """
    questions = [
        InsightQuestion(
            id="video_performance",
            category="Performance Analysis",
            question="Which video performed best and why?",
            description="Analyze overall video performance across all personas",
            requires_videos=False
        ),
        InsightQuestion(
            id="persona_preferences",
            category="Audience Insights",
            question="What are the key persona preferences?",
            description="Understand what different persona segments prefer",
            requires_videos=False
        ),
        InsightQuestion(
            id="strengths_weaknesses",
            category="Video Analysis",
            question="What are each video's strengths and weaknesses?",
            description="Detailed analysis of what works and what doesn't",
            requires_videos=False
        ),
        InsightQuestion(
            id="synthesis_recommendation",
            category="Optimization",
            question="How should I combine videos for best results?",
            description="Get recommendations for creating an optimal synthesis video",
            requires_videos=False
        ),
        InsightQuestion(
            id="visual_analysis",
            category="Visual Content",
            question="Analyze the visual elements in my videos",
            description="Deep dive into colors, scenes, pacing, and visual storytelling",
            requires_videos=True
        ),
        InsightQuestion(
            id="comparison",
            category="Comparison",
            question="Compare these specific videos",
            description="Side-by-side comparison of selected videos",
            requires_videos=True
        ),
        InsightQuestion(
            id="psychology_principles",
            category="Psychology",
            question="What psychological principles make this effective?",
            description="Understand the psychology behind what works",
            requires_videos=False
        ),
        InsightQuestion(
            id="improvement_suggestions",
            category="Optimization",
            question="What specific improvements would you suggest?",
            description="Actionable recommendations for improvement",
            requires_videos=True
        )
    ]
    return questions


@router.post("/campaigns/{campaign_id}/generate-insights")
async def generate_insights(campaign_id: str, request: GenerateInsightsRequest):
    """
    Generate comprehensive structured insights using Gemini 2.5
    Returns structured data that maps to dynamic UI cards
    """
    try:
        # Verify campaign exists
        campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Get all data
        videos = list(db.videos.find({"campaign_id": campaign_id}))
        if not videos:
            raise HTTPException(status_code=404, detail="No videos found")

        # Use evaluations from request if provided, otherwise fetch from DB
        if request.evaluations:
            evaluations = request.evaluations
        else:
            evaluations = list(db.evaluations.find({"campaign_id": campaign_id}))

        # Also check simulations if no evaluations found
        if not evaluations and request.simulations:
            # Extract evaluations from completed simulations
            for sim in request.simulations:
                if sim.get('results') and sim['results'].get('evaluations'):
                    evaluations.extend(sim['results']['evaluations'])

        personas = list(db.personas.find({}))

        # Build comprehensive context
        context = {
            "campaign": {
                "name": campaign.get("name"),
                "description": campaign.get("description"),
                "target_audience": campaign.get("target_audience")
            },
            "videos": [{
                "number": idx,
                "title": video.get("title"),
                "analysis": video.get("analysis")
            } for idx, video in enumerate(videos, 1)],
            "evaluations": [{
                "persona_name": eval.get("persona_name"),
                "most_preferred": eval.get("evaluation", {}).get("most_preferred_video"),
                "ranking": eval.get("evaluation", {}).get("preference_ranking"),
                "opinions": eval.get("evaluation", {}).get("video_opinions"),
                "reasoning": eval.get("evaluation", {}).get("reasoning")
            } for eval in evaluations],
            "personas_count": len(personas)
        }

        # Generate structured insights with Gemini
        user_question = request.question or "Provide a comprehensive analysis of this campaign's performance"
        insights = await _generate_structured_insights(context, user_question)

        # Save insights to database
        insight_record = {
            "campaign_id": campaign_id,
            "question": user_question,
            "insights": insights,
            "evaluations_count": len(evaluations),
            "videos_count": len(videos),
            "created_at": datetime.utcnow(),
            "generated_by": "gemini-2.5-pro"
        }
        db.insights.insert_one(insight_record)

        return {
            "campaign_id": campaign_id,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating insights: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


async def _generate_structured_insights(context: Dict[str, Any], user_question: str = None) -> Dict[str, Any]:
    """Generate structured insights using Gemini 2.5 Pro"""
    model = genai.GenerativeModel('gemini-2.5-pro')

    user_question = user_question or "Provide a comprehensive analysis of this campaign's performance"

    prompt = f"""You are an expert video marketing analyst. Answer the user's question about this campaign using the data provided.

USER QUESTION: {user_question}

CAMPAIGN DATA:
- Campaign: {context['campaign']['name']}
- Videos: {len(context['videos'])} videos
- Evaluations: {len(context['evaluations'])} persona evaluations

DETAILED DATA:
{json.dumps(context, indent=2, default=str)}

IMPORTANT INSTRUCTIONS:
1. Answer the user's specific question directly and comprehensively
2. Return your answer as a JSON object with dynamic keys based on what's relevant to the question
3. Each key should be a section/topic relevant to answering the question
4. Each value should be a descriptive object with:
   - "title": A clear title for this section
   - "content": The main content (string, array, or object depending on what's appropriate)
   - "insights": Key takeaways as an array

EXAMPLE STRUCTURES (adapt based on the question):

For "Why does video 2 perform better?":
{{
  "performance_comparison": {{
    "title": "Video 2 vs Other Videos",
    "content": "Video 2 outperforms with 45% preference rate vs 20% average...",
    "insights": ["Key reason 1", "Key reason 2"]
  }},
  "key_differentiators": {{
    "title": "What Makes Video 2 Stand Out",
    "content": {{"hook": "...", "pacing": "...", "messaging": "..."}},
    "insights": ["Insight 1", "Insight 2"]
  }}
}}

For "Overall performance summary":
{{
  "executive_summary": {{
    "title": "Campaign Performance Overview",
    "content": "This campaign achieved...",
    "insights": ["Top video: #2 at 45%", "Average engagement: 3.2/5"]
  }},
  "video_performance": {{
    "title": "Individual Video Results",
    "content": [{{"video": 1, "score": "35%"}}, {{"video": 2, "score": "45%"}}],
    "insights": ["Video 2 leads", "Video 1 is close second"]
  }},
  "recommendations": {{
    "title": "Action Items",
    "content": "Based on the data...",
    "insights": ["Action 1", "Action 2"]
  }}
}}

GUIDELINES:
- Create sections that DIRECTLY answer the user's question
- Use clear, descriptive key names (e.g., "psychological_triggers", "audience_segments", "creative_analysis")
- Include data-driven insights, not generic observations
- Always include a "summary" section with title, content, and insights
- Be specific and actionable

Return ONLY valid JSON. The structure should vary based on the question asked."""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )

        insights = json.loads(response.text)
        return insights

    except Exception as e:
        print(f"Error in Gemini call: {e}")
        raise


async def _generate_video_analysis(video_number: int, video_data: Dict, opinions: List[Dict]) -> Dict:
    """Generate strengths and weaknesses for a single video"""
    model = genai.GenerativeModel('gemini-2.5-flash')  # Using flash for faster response

    opinions_text = "\n".join([
        f"- {op['persona']}: {op['opinion']}"
        for op in opinions
    ])

    prompt = f"""Analyze this marketing video and provide strengths and weaknesses.

VIDEO {video_number}:
Title: {video_data.get('title', 'Unknown')}
Analysis: {json.dumps(video_data.get('analysis', {}), default=str)}

PERSONA OPINIONS ({len(opinions)} samples):
{opinions_text if opinions_text else "No opinions available"}

Provide a JSON response with:
{{
  "videoNumber": {video_number},
  "title": "Video {video_number} Analysis",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"]
}}

List 3-5 specific, actionable strengths and weaknesses based on the data provided."""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)


@router.post("/insights/chat", response_model=InsightChatResponse)
async def insights_chat(message: InsightChatMessage):
    """
    Interactive insights chat - answer one question at a time
    Supports video analysis with Gemini 2.5 when videos are attached
    """
    try:
        # Get campaign data if provided
        campaign_data = None
        videos_data = []
        evaluations_data = []

        if message.campaign_id:
            campaign = db.campaigns.find_one({"_id": ObjectId(message.campaign_id)})
            if campaign:
                campaign_data = {
                    "name": campaign.get("name"),
                    "description": campaign.get("description"),
                    "target_audience": campaign.get("target_audience")
                }

                # Get videos and evaluations
                videos_data = list(db.videos.find({"campaign_id": message.campaign_id}))
                evaluations_data = list(db.evaluations.find({"campaign_id": message.campaign_id}))

        # Get cached videos or video URLs if provided
        video_files = []
        for video_id in message.video_ids:
            cached_video = get_cached_video(message.session_id, video_id)
            if cached_video:
                video_files.append(cached_video)
            else:
                # Try to get video from database
                video_doc = db.videos.find_one({"_id": ObjectId(video_id)})
                if video_doc and video_doc.get("url"):
                    # Upload video to Gemini and cache it
                    try:
                        video_file = genai.upload_file(video_doc["url"])
                        # Wait for processing
                        while video_file.state.name == "PROCESSING":
                            time.sleep(2)
                            video_file = genai.get_file(video_file.name)

                        if video_file.state.name == "ACTIVE":
                            cache_video(message.session_id, video_id, video_file)
                            video_files.append(video_file)
                    except Exception as e:
                        print(f"Error uploading video {video_id}: {e}")

        # Build context for Gemini
        context = _build_chat_context(
            question=message.question,
            campaign=campaign_data,
            videos=videos_data,
            evaluations=evaluations_data,
            video_files=video_files,
            additional_context=message.context
        )

        # Generate answer using Gemini 2.5
        answer = await _generate_chat_answer(context, video_files)

        return InsightChatResponse(
            answer=answer,
            session_id=message.session_id,
            cached_videos=message.video_ids,
            generated_at=datetime.utcnow()
        )

    except Exception as e:
        print(f"Error in insights chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@router.post("/insights/upload-video/{session_id}/{video_id}")
async def upload_video_to_cache(
    session_id: str,
    video_id: str,
    file: UploadFile = File(...)
):
    """
    Upload a video file and cache it for the session
    This allows videos to be analyzed with Gemini without re-uploading
    """
    try:
        # Check if already cached
        cached = get_cached_video(session_id, video_id)
        if cached:
            return {
                "message": "Video already cached",
                "session_id": session_id,
                "video_id": video_id,
                "cached": True
            }

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Upload to Gemini
            video_file = genai.upload_file(tmp_path)

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "ACTIVE":
                # Cache the video
                cache_video(session_id, video_id, video_file)

                return {
                    "message": "Video uploaded and cached successfully",
                    "session_id": session_id,
                    "video_id": video_id,
                    "cached": True,
                    "expires_in_hours": CACHE_TTL_HOURS
                }
            else:
                raise HTTPException(status_code=500, detail="Video processing failed")

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        print(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


@router.delete("/insights/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear all cached videos for a session
    Call this when user session ends
    """
    clear_session_cache(session_id)
    return {
        "message": "Session cache cleared",
        "session_id": session_id
    }


def _build_chat_context(
    question: str,
    campaign: Optional[Dict] = None,
    videos: List[Dict] = None,
    evaluations: List[Dict] = None,
    video_files: List[Any] = None,
    additional_context: Optional[Dict] = None
) -> str:
    """Build context for chat"""
    context_parts = []

    if campaign:
        context_parts.append(f"CAMPAIGN: {campaign['name']}")
        if campaign.get('description'):
            context_parts.append(f"Description: {campaign['description']}")

    if videos:
        context_parts.append(f"\nVIDEOS ({len(videos)} total):")
        for idx, video in enumerate(videos, 1):
            context_parts.append(f"\nVideo {idx}:")
            context_parts.append(f"- Title: {video.get('title', 'Unknown')}")
            if video.get('analysis'):
                context_parts.append(f"- Analysis: {json.dumps(video['analysis'], default=str)[:500]}")

    if evaluations:
        summary = _summarize_evaluations(evaluations, videos or [])
        context_parts.append(f"\nEVALUATIONS:")
        context_parts.append(f"- Total evaluations: {summary['total_evaluations']}")
        context_parts.append(f"- Performance: {summary['video_votes']}")

    if additional_context:
        context_parts.append(f"\nADDITIONAL CONTEXT:")
        context_parts.append(json.dumps(additional_context, indent=2, default=str))

    context_parts.append(f"\nQUESTION: {question}")

    return "\n".join(context_parts)


async def _generate_chat_answer(context: str, video_files: List[Any] = None) -> str:
    """Generate answer using Gemini 2.5"""
    model = genai.GenerativeModel('gemini-2.5-pro')

    prompt = f"""You are an expert video marketing analyst and insights generator.
Answer the following question based on the provided context and video analysis (if videos are attached).

{context}

Provide a detailed, actionable answer. If videos are attached, analyze their visual content, pacing,
messaging, and effectiveness. Be specific and data-driven in your response."""

    # Include video files if provided
    content_parts = [prompt]
    if video_files:
        content_parts.extend(video_files)

    response = model.generate_content(
        content_parts,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7
        )
    )

    return response.text


def _summarize_evaluations(evaluations: List[Dict], videos: List[Dict]) -> Dict[str, Any]:
    """Summarize evaluation data for Gemini"""
    video_votes = {}
    video_rankings = {}

    for video in videos:
        video_id = str(video["_id"])
        video_votes[video_id] = 0
        video_rankings[video_id] = {"1st": 0, "2nd": 0, "3rd": 0, "4th": 0}

    for eval in evaluations:
        # Count most preferred votes
        preferred = eval.get("evaluation", {}).get("most_preferred_video")
        if preferred and preferred in video_votes:
            video_votes[preferred] += 1

        # Count rankings
        ranking = eval.get("evaluation", {}).get("preference_ranking", [])
        for idx, video_id in enumerate(ranking):
            if video_id in video_rankings:
                rank_keys = ["1st", "2nd", "3rd", "4th"]
                if idx < len(rank_keys):
                    video_rankings[video_id][rank_keys[idx]] += 1

    total_evals = len(evaluations)

    return {
        "total_evaluations": total_evals,
        "video_votes": video_votes,
        "video_rankings": video_rankings,
        "sample_opinions": _get_sample_opinions(evaluations)
    }


def _get_sample_opinions(evaluations: List[Dict], sample_size: int = 5) -> List[Dict]:
    """Get sample opinions from evaluations"""
    opinions = []
    for eval in evaluations[:sample_size]:
        video_opinions = eval.get("evaluation", {}).get("video_opinions", {})
        reasoning = eval.get("evaluation", {}).get("reasoning", "")
        opinions.append({
            "persona": eval.get("persona_name"),
            "video_opinions": video_opinions,
            "reasoning": reasoning
        })
    return opinions


@router.get("/campaigns/{campaign_id}/insights")
async def get_campaign_insights(campaign_id: str):
    """Get all saved insights for a campaign"""
    try:
        # Get all insights for this campaign, sorted by most recent first
        insights = list(db.insights.find(
            {"campaign_id": campaign_id}
        ).sort("created_at", -1))

        # Convert ObjectId to string and format response
        result = []
        for insight in insights:
            result.append({
                "id": str(insight["_id"]),
                "campaign_id": insight["campaign_id"],
                "question": insight.get("question", "General analysis"),
                "insights": insight.get("insights", {}),
                "evaluations_count": insight.get("evaluations_count", 0),
                "videos_count": insight.get("videos_count", 0),
                "created_at": insight.get("created_at", datetime.utcnow()).isoformat(),
                "generated_by": insight.get("generated_by", "unknown")
            })

        return result

    except Exception as e:
        print(f"Error retrieving insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


