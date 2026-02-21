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
import httpx
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/instagram", tags=["instagram"])


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
        "displayUrl": doc.get("displayUrl"),
        "images": doc.get("images", []),
        "videoUrl": doc.get("videoUrl"),
        "videoViewCount": doc.get("videoViewCount", 0),
        "videoPlayCount": doc.get("videoPlayCount", 0),
        "dimensionsHeight": doc.get("dimensionsHeight"),
        "dimensionsWidth": doc.get("dimensionsWidth"),
        "ownerUsername": doc.get("ownerUsername"),
        "ownerFullName": doc.get("ownerFullName"),
        "childPosts": doc.get("childPosts"),
    }


def reel_helper(doc) -> dict:
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
        "displayUrl": doc.get("displayUrl"),
        "images": doc.get("images", []),
        "videoUrl": doc.get("videoUrl"),
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
