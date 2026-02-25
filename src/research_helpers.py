#!/usr/bin/env python3
"""
Research helpers — pure logic for engagement scoring, top performers,
trend aggregation, brand URL analysis, financial data, and video AI analysis.
"""

import os
import json
import time
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Ensure TTL index on research_cache
db.research_cache.create_index("expires_at", expireAfterSeconds=0)


# ---------------------------------------------------------------------------
# Engagement scoring
# ---------------------------------------------------------------------------

def compute_engagement_score(
    post: Dict[str, Any],
    followers_count: int,
) -> float:
    """
    Composite engagement score:
      base = (likes + comments) / followers * 100
      + view bonus (if video)
      + recency bonus (posts < 30 days get up to +1.0)
    """
    if followers_count <= 0:
        return 0.0

    likes = post.get("likesCount", 0) or 0
    comments = post.get("commentsCount", 0) or 0
    views = post.get("videoPlayCount", 0) or post.get("videoViewCount", 0) or 0

    base = (likes + comments) / followers_count * 100

    # View bonus: if views > followers, add proportional bonus (capped at +2)
    view_bonus = 0.0
    if views > 0 and followers_count > 0:
        view_ratio = views / followers_count
        view_bonus = min(view_ratio * 0.5, 2.0)

    # Recency bonus: posts from last 30 days get up to +1.0
    recency_bonus = 0.0
    timestamp = post.get("timestamp")
    if timestamp:
        try:
            post_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            days_ago = (datetime.now(post_date.tzinfo) - post_date).days
            if days_ago < 30:
                recency_bonus = 1.0 * (1 - days_ago / 30)
        except (ValueError, TypeError):
            pass

    return round(base + view_bonus + recency_bonus, 4)


# ---------------------------------------------------------------------------
# Top performers
# ---------------------------------------------------------------------------

def get_top_performers(
    username: str,
    percentile: float = 0.2,
) -> Dict[str, Any]:
    """
    Query instagram_reels for username, score them,
    and return the top `percentile` fraction.
    """
    profile = db.instagram_profiles.find_one({"username": username})
    if not profile:
        return {"error": f"Profile not found for @{username}"}

    followers = profile.get("followersCount", 0) or 0

    reels = list(db.instagram_reels.find({"ownerUsername": username}))
    for r in reels:
        r["_content_type"] = "reel"

    all_content: List[Dict[str, Any]] = reels

    if not all_content:
        return {
            "username": username,
            "followers": followers,
            "total_reels": 0,
            "top_performers": [],
        }

    # Score everything
    for item in all_content:
        item["engagement_score"] = compute_engagement_score(item, followers)

    # Sort descending
    all_content.sort(key=lambda x: x["engagement_score"], reverse=True)

    # Top percentile
    top_n = max(1, int(len(all_content) * percentile))
    top_items = all_content[:top_n]

    def _serialize(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(item.get("_id", "")),
            "shortCode": item.get("shortCode"),
            "engagement_score": item["engagement_score"],
            "likesCount": item.get("likesCount", 0),
            "commentsCount": item.get("commentsCount", 0),
            "videoPlayCount": item.get("videoPlayCount", 0) or item.get("videoViewCount", 0),
            "videoDuration": item.get("videoDuration"),
            "caption": (item.get("caption") or "")[:300],
            "displayUrl": item.get("displayUrl"),
            "videoUrl": item.get("videoUrl"),
            "timestamp": item.get("timestamp"),
            "hashtags": item.get("hashtags", []),
            "url": item.get("url"),
        }

    return {
        "username": username,
        "followers": followers,
        "total_reels": len(all_content),
        "top_count": len(top_items),
        "top_performers": [_serialize(item) for item in top_items],
    }


# ---------------------------------------------------------------------------
# Trend data (time-series)
# ---------------------------------------------------------------------------

def build_trend_data(username: str) -> List[Dict[str, Any]]:
    """
    Group reels by date and aggregate likes, comments, views.
    Returns flat time-series list for charting.
    """
    docs = list(db.instagram_reels.find({"ownerUsername": username}))
    date_map: Dict[str, Dict[str, int]] = {}
    for doc in docs:
        ts = doc.get("timestamp")
        if not ts:
            continue
        try:
            date_str = ts[:10]  # "YYYY-MM-DD"
        except (TypeError, IndexError):
            continue

        if date_str not in date_map:
            date_map[date_str] = {"likes": 0, "comments": 0, "views": 0, "count": 0}

        date_map[date_str]["likes"] += doc.get("likesCount", 0) or 0
        date_map[date_str]["comments"] += doc.get("commentsCount", 0) or 0
        date_map[date_str]["views"] += (doc.get("videoPlayCount", 0) or doc.get("videoViewCount", 0) or 0)
        date_map[date_str]["count"] += 1

    sorted_dates = sorted(date_map.keys())
    return [{"date": d, **date_map[d]} for d in sorted_dates]


# ---------------------------------------------------------------------------
# Brand URL analysis
# ---------------------------------------------------------------------------

def extract_brand_from_url(url: str) -> Dict[str, Any]:
    """
    Fetch URL with httpx, parse HTML, use LLM to extract brand insights.
    Results are cached in research_cache collection.
    """
    cache_key = f"brand_url:{url}"
    cached = db.research_cache.find_one({"cache_key": cache_key})
    if cached and cached.get("expires_at", datetime.min) > datetime.utcnow():
        return cached["data"]

    from bs4 import BeautifulSoup
    from src.research_prompts import BRAND_URL_ANALYSIS_PROMPT

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30.0, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()

    text_content = soup.get_text(separator="\n", strip=True)
    # Trim to ~8000 chars for LLM context
    text_content = text_content[:8000]

    prompt = BRAND_URL_ANALYSIS_PROMPT.format(url=url, html_content=text_content)

    analysis = _call_llm(prompt)
    if analysis:
        analysis["url"] = url
        # Cache for 7 days
        db.research_cache.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "data": analysis,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=7),
            }},
            upsert=True,
        )
    return analysis or {"error": "LLM analysis failed"}


# ---------------------------------------------------------------------------
# Financial data
# ---------------------------------------------------------------------------

def fetch_financial_data(company_name: str) -> Dict[str, Any]:
    """
    Fetch financial data via web search: uses httpx to hit public sources,
    parses HTML, sends to LLM to extract metrics. Caches in research_cache.
    """
    cache_key = f"financial:{company_name.lower().replace(' ', '_')}"
    cached = db.research_cache.find_one({"cache_key": cache_key})
    if cached and cached.get("expires_at", datetime.min) > datetime.utcnow():
        return cached["data"]

    from bs4 import BeautifulSoup
    from src.research_prompts import FINANCIAL_EXTRACTION_PROMPT

    # Try multiple public sources
    search_urls = [
        f"https://finance.yahoo.com/quote/{_guess_ticker(company_name)}/",
        f"https://www.google.com/search?q={company_name.replace(' ', '+')}+revenue+subscribers+2024+2025",
    ]

    combined_content = ""
    for search_url in search_urls:
        try:
            resp = httpx.get(search_url, follow_redirects=True, timeout=20.0, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            })
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                combined_content += soup.get_text(separator="\n", strip=True)[:4000] + "\n\n---\n\n"
        except Exception:
            continue

    if not combined_content.strip():
        return {"error": f"Could not fetch financial data for {company_name}"}

    prompt = FINANCIAL_EXTRACTION_PROMPT.format(
        company_name=company_name,
        web_content=combined_content[:8000],
    )

    analysis = _call_llm(prompt)
    if analysis:
        analysis["company_name"] = company_name
        db.research_cache.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "data": analysis,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=7),
            }},
            upsert=True,
        )
    return analysis or {"error": "LLM extraction failed"}


def _guess_ticker(company_name: str) -> str:
    """Map known company names to stock tickers."""
    tickers = {
        "stitchfix": "SFIX",
        "stitch fix": "SFIX",
        "renttherunway": "RENT",
        "rent the runway": "RENT",
    }
    return tickers.get(company_name.lower(), company_name.upper()[:4])


# ---------------------------------------------------------------------------
# Video AI analysis (Gemini)
# ---------------------------------------------------------------------------

def analyze_video_content(video_url: str, proxy: bool = True) -> Dict[str, Any]:
    """
    Download video (via proxy if Instagram CDN), upload to Gemini,
    and extract detailed content analysis.
    """
    import google.generativeai as genai
    from src.research_prompts import VIDEO_UNDERSTANDING_PROMPT

    cache_key = f"video_ai:{video_url[:200]}"
    cached = db.research_cache.find_one({"cache_key": cache_key})
    if cached and cached.get("expires_at", datetime.min) > datetime.utcnow():
        return cached["data"]

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY not configured"}
    genai.configure(api_key=GOOGLE_API_KEY)

    # Download video to temp file
    download_url = video_url
    if proxy and ("cdninstagram.com" in video_url or "fbcdn.net" in video_url):
        # Use our own proxy endpoint base URL
        api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
        download_url = f"{api_base}/api/instagram/proxy?url={video_url}"

    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as http_client:
            resp = http_client.get(download_url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            })
            resp.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to download video: {str(e)}"}

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        # Upload to Gemini
        video_file = genai.upload_file(path=tmp_path, mime_type="video/mp4")

        # Wait for processing
        max_wait = 120
        waited = 0
        while video_file.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            return {"error": f"Video processing failed: state={video_file.state.name}"}

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([video_file, VIDEO_UNDERSTANDING_PROMPT])

        # Parse JSON response
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        analysis = json.loads(text.strip())

        # Cache for 7 days
        db.research_cache.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "data": analysis,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(days=7),
            }},
            upsert=True,
        )

        return analysis

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse Gemini response: {str(e)}"}
    except Exception as e:
        return {"error": f"Video analysis failed: {str(e)}"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Frame extraction + GCS upload
# ---------------------------------------------------------------------------

def _get_gcs_bucket():
    """Reuse GCS init pattern from brand_assets."""
    from google.cloud import storage
    from google.oauth2 import service_account

    GCS_BUCKET = os.getenv("GCS_BUCKET", "dble-input-videos")
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        creds_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
    else:
        credentials = None
        storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    return bucket, credentials, GCS_BUCKET


def _get_signed_url(bucket, blob_name: str, credentials) -> str:
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET",
        credentials=credentials,
    )


def extract_reel_frames(
    video_url: str,
    short_code: str,
    username: str,
    num_frames: int = 4,
) -> List[Dict[str, Any]]:
    """
    Download a reel, extract key frames with ffmpeg, upload to GCS.
    Returns list of {frame_index, gcs_blob_name, signed_url, timestamp_sec}.
    """
    import subprocess

    # Download video directly (no proxy needed server-side)
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as http_client:
            resp = http_client.get(video_url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            })
            resp.raise_for_status()
    except Exception as e:
        return [{"error": f"Download failed: {e}"}]

    tmpdir = tempfile.mkdtemp(prefix="research_frames_")
    video_path = os.path.join(tmpdir, f"{short_code}.mp4")
    with open(video_path, "wb") as f:
        f.write(resp.content)

    try:
        # Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0

        # Extract frames at evenly spaced intervals
        frames: List[Dict[str, Any]] = []
        bucket, credentials, bucket_name = _get_gcs_bucket()

        for i in range(num_frames):
            ts = (duration / (num_frames + 1)) * (i + 1)
            frame_path = os.path.join(tmpdir, f"frame_{i}.jpg")

            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", frame_path],
                capture_output=True, timeout=30,
            )

            if not os.path.exists(frame_path):
                continue

            # Upload to GCS
            blob_name = f"research-assets/{username}/{short_code}/frame_{i}.jpg"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(frame_path, content_type="image/jpeg")

            signed_url = _get_signed_url(bucket, blob_name, credentials)

            frames.append({
                "frame_index": i,
                "timestamp_sec": round(ts, 2),
                "gcs_blob_name": blob_name,
                "gs_uri": f"gs://{bucket_name}/{blob_name}",
                "signed_url": signed_url,
            })

        return frames

    except Exception as e:
        return [{"error": f"Frame extraction failed: {e}"}]
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def extract_assets_for_top_reels(
    username: str,
    top_performers: List[Dict[str, Any]],
    max_reels: int = 10,
    frames_per_reel: int = 4,
) -> List[Dict[str, Any]]:
    """
    Extract frames from top reels and upload to GCS.
    Returns updated top_performers with 'extracted_frames' field.
    """
    results = []
    reels_with_video = [p for p in top_performers if p.get("videoUrl")][:max_reels]

    for reel in reels_with_video:
        short_code = reel.get("shortCode", "unknown")
        print(f"[research] Extracting frames from @{username}/{short_code}...")
        frames = extract_reel_frames(
            video_url=reel["videoUrl"],
            short_code=short_code,
            username=username,
            num_frames=frames_per_reel,
        )
        reel_copy = dict(reel)
        reel_copy["extracted_frames"] = frames
        results.append(reel_copy)

    return results


# ---------------------------------------------------------------------------
# LLM helper (uses Anthropic Claude by default)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call Claude to get structured JSON output."""
    try:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())
    except Exception as e:
        print(f"[research_helpers] LLM call failed: {e}")
        return None
