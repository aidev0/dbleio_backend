#!/usr/bin/env python3
"""
Instagram API — profiles, posts, reels (read) + Apify scrape (write).
Collections: instagram_profiles, instagram_posts, instagram_reels
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
import httpx
from datetime import timedelta
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/instagram", tags=["instagram"])

GCS_BUCKET = os.getenv("GCS_BUCKET", "dble-input-videos")


# --- GCS signed URL helper ---

def _get_signed_url(gs_uri: str) -> Optional[str]:
    """Generate a fresh signed URL from a gs:// URI. Returns None on failure."""
    if not gs_uri or not gs_uri.startswith("gs://"):
        return None
    try:
        from google.cloud import storage
        from google.oauth2 import service_account

        creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if creds_json:
            creds_info = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
        else:
            credentials = None
            storage_client = storage.Client()

        blob_path = gs_uri.replace(f"gs://{GCS_BUCKET}/", "")
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_path)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET",
            credentials=credentials,
        )
    except Exception:
        return None


def _resolve_url(gs_uri: Optional[str], meta_url: Optional[str]) -> Optional[str]:
    """Return signed GCS URL if available, otherwise fall back to Meta URL."""
    if gs_uri:
        signed = _get_signed_url(gs_uri)
        if signed:
            return signed
    return meta_url


# --- Helpers ---

def profile_helper(doc) -> dict:
    return {
        "_id": str(doc["_id"]),
        "username": doc.get("username"),
        "fullName": doc.get("fullName"),
        "profilePicUrl": doc.get("profilePicUrl"),
        "postsCount": doc.get("postsCount"),
        "followersCount": doc.get("followersCount"),
        "followsCount": doc.get("followsCount"),
        "biography": doc.get("biography"),
        "verified": doc.get("verified", False),
        "isBusinessAccount": doc.get("isBusinessAccount", False),
        "private": doc.get("private", False),
    }


def post_helper(doc) -> dict:
    # Resolve images: GCS first, then Meta fallback
    gcs_images = doc.get("gcs_images", [])
    meta_images = doc.get("images", [])
    resolved_images = []
    for i, gs_uri in enumerate(gcs_images):
        resolved_images.append(_resolve_url(gs_uri, meta_images[i] if i < len(meta_images) else None))
    # Include any remaining Meta images not covered by GCS
    for i in range(len(gcs_images), len(meta_images)):
        resolved_images.append(meta_images[i])

    return {
        "_id": str(doc["_id"]),
        "id": doc.get("id"),
        "type": doc.get("type"),
        "shortCode": doc.get("shortCode"),
        "caption": doc.get("caption"),
        "hashtags": doc.get("hashtags", []),
        "url": doc.get("url"),
        "commentsCount": doc.get("commentsCount", 0),
        "likesCount": doc.get("likesCount", 0),
        "timestamp": doc.get("timestamp"),
        "displayUrl": _resolve_url(doc.get("gcs_display_uri"), doc.get("displayUrl")),
        "images": resolved_images,
        "videoUrl": _resolve_url(doc.get("gcs_video_uri"), doc.get("videoUrl")),
        "audioUrl": _resolve_url(doc.get("gcs_audio_uri"), doc.get("audioUrl")),
        "videoViewCount": doc.get("videoViewCount", 0),
        "videoPlayCount": doc.get("videoPlayCount", 0),
        "dimensionsHeight": doc.get("dimensionsHeight"),
        "dimensionsWidth": doc.get("dimensionsWidth"),
        "ownerUsername": doc.get("ownerUsername"),
        "ownerFullName": doc.get("ownerFullName"),
        "childPosts": doc.get("childPosts"),
        "ai_analysis": doc.get("ai_analysis"),
    }


def reel_helper(doc) -> dict:
    gcs_images = doc.get("gcs_images", [])
    meta_images = doc.get("images", [])
    resolved_images = []
    for i, gs_uri in enumerate(gcs_images):
        resolved_images.append(_resolve_url(gs_uri, meta_images[i] if i < len(meta_images) else None))
    for i in range(len(gcs_images), len(meta_images)):
        resolved_images.append(meta_images[i])

    return {
        "_id": str(doc["_id"]),
        "id": doc.get("id"),
        "type": doc.get("type"),
        "shortCode": doc.get("shortCode"),
        "caption": doc.get("caption"),
        "hashtags": doc.get("hashtags", []),
        "url": doc.get("url"),
        "commentsCount": doc.get("commentsCount", 0),
        "likesCount": doc.get("likesCount", 0),
        "timestamp": doc.get("timestamp"),
        "displayUrl": _resolve_url(doc.get("gcs_display_uri"), doc.get("displayUrl")),
        "images": resolved_images,
        "videoUrl": _resolve_url(doc.get("gcs_video_uri"), doc.get("videoUrl")),
        "audioUrl": _resolve_url(doc.get("gcs_audio_uri"), doc.get("audioUrl")),
        "videoDuration": doc.get("videoDuration"),
        "videoViewCount": doc.get("videoViewCount", 0),
        "videoPlayCount": doc.get("videoPlayCount", 0),
        "dimensionsHeight": doc.get("dimensionsHeight"),
        "dimensionsWidth": doc.get("dimensionsWidth"),
        "ownerUsername": doc.get("ownerUsername"),
        "ownerFullName": doc.get("ownerFullName"),
    }


# --- Read endpoints ---

@router.get("/profiles")
async def list_profiles(username: Optional[str] = None):
    """Return all profiles, or filter by username."""
    query = {}
    if username:
        query["username"] = username
    docs = list(db.instagram_profiles.find(query).sort("username", 1))
    return [profile_helper(d) for d in docs]


@router.get("/posts")
async def list_posts(username: Optional[str] = None, skip: int = 0, limit: int = 12):
    """Paginated posts, sorted by timestamp desc."""
    query = {}
    if username:
        query["ownerUsername"] = username
    total = db.instagram_posts.count_documents(query)
    docs = list(
        db.instagram_posts.find(query)
        .sort("timestamp", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    return {"items": [post_helper(d) for d in docs], "total": total, "skip": skip, "limit": limit}


@router.get("/reels")
async def list_reels(username: Optional[str] = None, skip: int = 0, limit: int = 12):
    """Paginated reels, sorted by timestamp desc."""
    query = {}
    if username:
        query["ownerUsername"] = username
    total = db.instagram_reels.count_documents(query)
    docs = list(
        db.instagram_reels.find(query)
        .sort("timestamp", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    return {"items": [reel_helper(d) for d in docs], "total": total, "skip": skip, "limit": limit}


# --- Scrape ---

class ScrapeRequest(BaseModel):
    instagram_url: str


def _extract_username(url: str) -> str:
    """Pull username from an Instagram URL like https://www.instagram.com/humansofny/"""
    match = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", url)
    if not match:
        raise ValueError(f"Could not extract username from URL: {url}")
    return match.group(1)


def _download_media(url: str) -> Optional[bytes]:
    """Download media from a URL, return bytes or None."""
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as c:
            resp = c.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            })
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        print(f"[instagram] Download failed: {e}")
        return None


def _upload_to_gcs(blob_path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to GCS bucket, return gs:// URI."""
    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account

    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if creds_json:
        creds_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        client = gcs_storage.Client(credentials=credentials, project=credentials.project_id)
    else:
        client = gcs_storage.Client()

    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{GCS_BUCKET}/{blob_path}"


def _upload_media_for_collection(collection, username: str):
    """Download media from Meta CDN and upload to GCS for all docs in a collection."""
    docs = list(collection.find({"ownerUsername": username}))
    print(f"[instagram] Uploading media for {len(docs)} docs (@{username})")

    for doc in docs:
        short_code = doc.get("shortCode", str(doc["_id"]))
        prefix = f"instagram/{username}/{short_code}"
        updates = {}

        # displayUrl → image
        display_url = doc.get("displayUrl")
        if display_url and not doc.get("gcs_display_uri"):
            data = _download_media(display_url)
            if data:
                updates["gcs_display_uri"] = _upload_to_gcs(f"{prefix}/display.jpg", data, "image/jpeg")

        # videoUrl → video
        video_url = doc.get("videoUrl")
        if video_url and not doc.get("gcs_video_uri"):
            data = _download_media(video_url)
            if data:
                updates["gcs_video_uri"] = _upload_to_gcs(f"{prefix}/video.mp4", data, "video/mp4")

        # audioUrl → audio
        audio_url = doc.get("audioUrl")
        if audio_url and not doc.get("gcs_audio_uri"):
            data = _download_media(audio_url)
            if data:
                updates["gcs_audio_uri"] = _upload_to_gcs(f"{prefix}/audio.mp4", data, "audio/mp4")

        # images[] → individual images
        images = doc.get("images", [])
        if images and not doc.get("gcs_images"):
            gcs_images = []
            for i, img_url in enumerate(images):
                if isinstance(img_url, str) and img_url:
                    data = _download_media(img_url)
                    if data:
                        gcs_images.append(_upload_to_gcs(f"{prefix}/image_{i}.jpg", data, "image/jpeg"))
            if gcs_images:
                updates["gcs_images"] = gcs_images

        if updates:
            collection.update_one({"_id": doc["_id"]}, {"$set": updates})
            print(f"[instagram] Uploaded {len(updates)} media for {short_code}")


def _run_scrape(username: str):
    """Background task: call Apify actors and insert results into MongoDB."""
    from apify_client import ApifyClient

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        print("APIFY_API_TOKEN not set, skipping scrape")
        return

    apify = ApifyClient(token)
    profile_url = f"https://www.instagram.com/{username}/"

    # 1) Posts
    print(f"[instagram] Scraping posts for @{username} …")
    posts_run = apify.actor("shu8hvrXbJbY3Eb9W").call(
        run_input={
            "directUrls": [profile_url],
            "resultsType": "posts",
            "resultsLimit": 100,
            "onlyPostsNewerThan": None,
            "search": None,
            "searchType": "hashtag",
            "searchLimit": 1,
            "addParentData": False,
        }
    )
    posts_items = list(apify.dataset(posts_run["defaultDatasetId"]).iterate_items())
    if posts_items:
        db.instagram_posts.delete_many({"ownerUsername": username})
        db.instagram_posts.insert_many(posts_items)
    print(f"[instagram] Inserted {len(posts_items)} posts for @{username}")

    # Upload post media to GCS
    _upload_media_for_collection(db.instagram_posts, username)

    # 2) Reels
    print(f"[instagram] Scraping reels for @{username} …")
    reels_run = apify.actor("shu8hvrXbJbY3Eb9W").call(
        run_input={
            "directUrls": [profile_url],
            "resultsType": "reels",
            "resultsLimit": 100,
            "onlyPostsNewerThan": None,
            "search": None,
            "searchType": "hashtag",
            "searchLimit": 1,
            "addParentData": False,
        }
    )
    reels_items = list(apify.dataset(reels_run["defaultDatasetId"]).iterate_items())
    if reels_items:
        db.instagram_reels.delete_many({"ownerUsername": username})
        db.instagram_reels.insert_many(reels_items)
    print(f"[instagram] Inserted {len(reels_items)} reels for @{username}")

    # Upload reel media to GCS
    _upload_media_for_collection(db.instagram_reels, username)

    # 3) Profile
    print(f"[instagram] Scraping profile for @{username} …")
    profile_run = apify.actor("dSCLg0C3YEZ83HzYX").call(
        run_input={"usernames": [username]}
    )
    profile_items = list(apify.dataset(profile_run["defaultDatasetId"]).iterate_items())
    if profile_items:
        db.instagram_profiles.delete_many({"username": username})
        db.instagram_profiles.insert_many(profile_items)
    print(f"[instagram] Inserted {len(profile_items)} profile(s) for @{username}")

    print(f"[instagram] Scrape complete for @{username}")


@router.post("/scrape")
async def scrape_account(body: ScrapeRequest, background_tasks: BackgroundTasks):
    """Kick off an Apify scrape in the background."""
    try:
        username = _extract_username(body.instagram_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(_run_scrape, username)
    return {"status": "scraping", "username": username}


@router.get("/scrape-status")
async def scrape_status(username: str):
    """Check whether data exists for a given username."""
    has_profile = db.instagram_profiles.count_documents({"username": username}) > 0
    has_posts = db.instagram_posts.count_documents({"ownerUsername": username}) > 0
    has_reels = db.instagram_reels.count_documents({"ownerUsername": username}) > 0
    return {
        "username": username,
        "ready": has_profile and has_posts and has_reels,
        "has_profile": has_profile,
        "has_posts": has_posts,
        "has_reels": has_reels,
    }


# --- Top posts by likes ---

@router.get("/posts/top")
async def top_posts(
    username: str,
    limit: int = 4,
    content_type: Optional[str] = None,
    sort: str = "desc",
):
    """
    Posts sorted by likesCount.
    content_type: 'post' (Image + Sidecar), 'reel' (Video), or omit for all.
    sort: 'desc' (most likes) or 'asc' (fewest likes).
    """
    query: dict = {"ownerUsername": username, "likesCount": {"$gt": 0}}
    if content_type == "post":
        query["type"] = {"$in": ["Image", "Sidecar"]}
    elif content_type == "reel":
        query["type"] = "Video"

    from pymongo import ASCENDING as ASC
    direction = ASC if sort == "asc" else DESCENDING

    docs = list(
        db.instagram_posts.find(query)
        .sort("likesCount", direction)
        .limit(limit)
    )
    return [post_helper(d) for d in docs]


@router.get("/posts/analyzed")
async def analyzed_posts(
    username: str,
    content_type: Optional[str] = None,
):
    """Return all posts that have ai_analysis, sorted by likesCount desc."""
    query: dict = {
        "ownerUsername": username,
        "ai_analysis": {"$ne": None},
    }
    if content_type == "post":
        query["type"] = {"$in": ["Image", "Sidecar"]}
    elif content_type == "reel":
        query["type"] = "Video"

    docs = list(
        db.instagram_posts.find(query).sort("likesCount", DESCENDING)
    )
    return [post_helper(d) for d in docs]


# --- Engagement stats (lightweight, for charting) ---

@router.get("/posts/engagement")
async def posts_engagement(username: str, content_type: Optional[str] = None):
    """
    Raw engagement data for charting.
    content_type: 'post' (Image + Sidecar), 'reel' (Video), or omit for all.
    """
    query: dict = {"ownerUsername": username}
    if content_type == "post":
        query["type"] = {"$in": ["Image", "Sidecar"]}
    elif content_type == "reel":
        query["type"] = "Video"

    docs = list(
        db.instagram_posts.find(
            query,
            {
                "timestamp": 1,
                "likesCount": 1,
                "commentsCount": 1,
                "videoPlayCount": 1,
                "videoViewCount": 1,
                "type": 1,
            },
        ).sort("timestamp", 1)
    )

    return [
        {
            "timestamp": d.get("timestamp"),
            "likesCount": d.get("likesCount", 0) or 0,
            "commentsCount": d.get("commentsCount", 0) or 0,
            "views": (d.get("videoPlayCount", 0) or d.get("videoViewCount", 0) or 0),
            "type": d.get("type"),
        }
        for d in docs
    ]


# --- Batch AI analysis of top posts ---

class AnalyzeTopRequest(BaseModel):
    brand_username: str
    competitor_username: str
    limit: int = 4

@router.post("/posts/analyze-top")
async def analyze_top_posts(body: AnalyzeTopRequest, background_tasks: BackgroundTasks):
    """
    Run AI analysis on top posts for brand + competitor, store results on documents,
    then run idea extraction across all analyses.
    """
    from bson import ObjectId as ObjId

    def _run(brand_username: str, competitor_username: str, limit: int):
        from src.research_helpers import analyze_video_content, analyze_image_content
        from src.research_prompts import IDEA_EXTRACTION_PROMPT
        import anthropic

        def _analyze_posts(username: str, n: int):
            query = {"ownerUsername": username, "likesCount": {"$gt": 0}}
            docs = list(db.instagram_posts.find(query).sort("likesCount", DESCENDING).limit(n))
            results = []
            for doc in docs:
                post_type = doc.get("type", "")
                existing = doc.get("ai_analysis")
                if existing and not existing.get("error"):
                    results.append({"shortCode": doc.get("shortCode"), "type": post_type, "analysis": existing})
                    continue

                if post_type == "Video":
                    # Prefer GCS signed URL over expired Meta CDN URL
                    gcs_uri = doc.get("gcs_video_uri")
                    video_url = _get_signed_url(gcs_uri) if gcs_uri else doc.get("videoUrl")
                    if not video_url:
                        results.append({"shortCode": doc.get("shortCode"), "type": post_type, "analysis": {"error": "no video URL"}})
                        continue
                    analysis = analyze_video_content(video_url, proxy=not gcs_uri)
                else:
                    # Image or Sidecar — prefer GCS signed URL
                    gcs_uri = doc.get("gcs_display_uri")
                    gcs_images = doc.get("gcs_images", [])
                    img_url = _get_signed_url(gcs_uri) if gcs_uri else None
                    if not img_url and gcs_images:
                        img_url = _get_signed_url(gcs_images[0])
                    if not img_url:
                        img_url = doc.get("displayUrl")
                    if not img_url:
                        imgs = doc.get("images", [])
                        img_url = imgs[0] if imgs else None
                    if not img_url:
                        results.append({"shortCode": doc.get("shortCode"), "type": post_type, "analysis": {"error": "no image URL"}})
                        continue
                    analysis = analyze_image_content(img_url)

                # Store on document
                db.instagram_posts.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"ai_analysis": analysis}},
                )
                results.append({"shortCode": doc.get("shortCode"), "type": post_type, "analysis": analysis})
            return results

        brand_results = _analyze_posts(brand_username, limit)
        comp_results = _analyze_posts(competitor_username, limit)

        # Run idea extraction
        def _fmt(results):
            parts = []
            for r in results:
                parts.append(f"Post {r['shortCode']} ({r['type']}):\n{json.dumps(r['analysis'], indent=2, default=str)}")
            return "\n\n".join(parts)

        try:
            client_anthropic = anthropic.Anthropic()
            prompt = IDEA_EXTRACTION_PROMPT.format(
                brand_username=brand_username,
                competitor_username=competitor_username,
                brand_analyses=_fmt(brand_results),
                competitor_analyses=_fmt(comp_results),
            )
            resp = client_anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            ideas = json.loads(text.strip())
        except Exception as e:
            ideas = {"error": str(e)}

        # Store ideas in research_cache for easy retrieval
        cache_key = f"ideas:{brand_username}:{competitor_username}"
        db.research_cache.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "data": {
                    "brand_analyses": brand_results,
                    "competitor_analyses": comp_results,
                    "ideas": ideas,
                },
                "created_at": __import__("datetime").datetime.utcnow(),
            }},
            upsert=True,
        )

    background_tasks.add_task(_run, body.brand_username, body.competitor_username, body.limit)
    return {"status": "started", "message": f"Analyzing top {body.limit} posts for @{body.brand_username} and @{body.competitor_username}"}


@router.get("/posts/ideas")
async def get_post_ideas(brand_username: str, competitor_username: str):
    """Get cached idea extraction results."""
    cache_key = f"ideas:{brand_username}:{competitor_username}"
    cached = db.research_cache.find_one({"cache_key": cache_key})
    if not cached:
        return {"status": "not_found"}
    return {"status": "ready", "data": cached.get("data", {}), "created_at": str(cached.get("created_at", ""))}


@router.get("/posts/detailed-ideas")
async def get_detailed_ideas(brand_username: str, competitor_username: str):
    """Get cached detailed production brief ideas."""
    cache_key = f"detailed_ideas:{brand_username}:{competitor_username}"
    cached = db.research_cache.find_one({"cache_key": cache_key})
    if not cached:
        return {"status": "not_found"}
    return {"status": "ready", "data": cached.get("data", {}), "created_at": str(cached.get("created_at", ""))}


# --- Image / video proxy (avoids CDN referrer blocks) ---

@router.get("/proxy")
async def proxy_media(url: str):
    """Proxy an Instagram CDN URL to avoid referrer-based blocking."""
    if not url or not ("cdninstagram.com" in url or "fbcdn.net" in url):
        raise HTTPException(status_code=400, detail="Only Instagram CDN URLs allowed")

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Upstream error")

        content_type = resp.headers.get("content-type", "application/octet-stream")
        return StreamingResponse(
            iter([resp.content]),
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
