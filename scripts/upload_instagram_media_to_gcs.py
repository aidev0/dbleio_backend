#!/usr/bin/env python3
"""
One-time script: Download media from instagram_posts docs and upload to GCS.
Stores gs_uri fields (gcs_display_uri, gcs_video_uri, gcs_audio_uri) on each doc.

Usage:
  cd /Users/jacobrafati/Projects/dble/dbleio_backend
  source venv/bin/activate
  python3 scripts/upload_instagram_media_to_gcs.py
"""

import os
import sys
import json
import httpx

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
from pymongo import MongoClient
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
GCS_BUCKET = os.getenv("GCS_BUCKET", "dble-input-videos")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# GCS setup
creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
creds_info = json.loads(creds_json)
credentials = service_account.Credentials.from_service_account_info(creds_info)
storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
bucket = storage_client.bucket(GCS_BUCKET)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def download(url: str) -> bytes | None:
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as c:
            resp = c.get(url, headers=HEADERS)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        print(f"    FAIL download: {e}")
        return None


def upload_to_gcs(blob_path: str, data: bytes, content_type: str) -> str:
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{GCS_BUCKET}/{blob_path}"


def process_post(doc):
    username = doc.get("ownerUsername", "unknown")
    short_code = doc.get("shortCode", str(doc["_id"]))
    prefix = f"instagram/{username}/{short_code}"
    updates = {}

    # 1) displayUrl → image
    display_url = doc.get("displayUrl")
    if display_url and not doc.get("gcs_display_uri"):
        print(f"  display: downloading...")
        data = download(display_url)
        if data:
            gs_uri = upload_to_gcs(f"{prefix}/display.jpg", data, "image/jpeg")
            updates["gcs_display_uri"] = gs_uri
            print(f"  display: {gs_uri}")

    # 2) videoUrl → video
    video_url = doc.get("videoUrl")
    if video_url and not doc.get("gcs_video_uri"):
        print(f"  video: downloading...")
        data = download(video_url)
        if data:
            gs_uri = upload_to_gcs(f"{prefix}/video.mp4", data, "video/mp4")
            updates["gcs_video_uri"] = gs_uri
            print(f"  video: {gs_uri}")

    # 3) audioUrl → audio
    audio_url = doc.get("audioUrl")
    if audio_url and not doc.get("gcs_audio_uri"):
        print(f"  audio: downloading...")
        data = download(audio_url)
        if data:
            gs_uri = upload_to_gcs(f"{prefix}/audio.mp4", data, "audio/mp4")
            updates["gcs_audio_uri"] = gs_uri
            print(f"  audio: {gs_uri}")

    # 4) images[] → individual images
    images = doc.get("images", [])
    if images and not doc.get("gcs_images"):
        gcs_images = []
        for i, img_url in enumerate(images):
            if isinstance(img_url, str) and img_url:
                print(f"  image[{i}]: downloading...")
                data = download(img_url)
                if data:
                    gs_uri = upload_to_gcs(f"{prefix}/image_{i}.jpg", data, "image/jpeg")
                    gcs_images.append(gs_uri)
                    print(f"  image[{i}]: {gs_uri}")
        if gcs_images:
            updates["gcs_images"] = gcs_images

    if updates:
        db.instagram_posts.update_one({"_id": doc["_id"]}, {"$set": updates})
        print(f"  saved {len(updates)} field(s)")
    else:
        print(f"  nothing to do (already uploaded or no media)")


def main():
    posts = list(db.instagram_posts.find())
    total = len(posts)
    print(f"Found {total} instagram_posts\n")

    for i, doc in enumerate(posts):
        username = doc.get("ownerUsername", "?")
        short_code = doc.get("shortCode", "?")
        print(f"[{i+1}/{total}] @{username} / {short_code}")
        process_post(doc)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
