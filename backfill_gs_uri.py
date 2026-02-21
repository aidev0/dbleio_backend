#!/usr/bin/env python3
"""
One-time migration: reconstruct gs_uri from existing signed GCS URLs in MongoDB.

Only works for videos already uploaded to GCS (OpenAI). Veo/Replicate videos
that were stored as provider temp URLs are unrecoverable — they must be
regenerated.

Usage:
    cd /Users/jacobrafati/Projects/dble/dbleio_backend
    source venv/bin/activate
    python3 backfill_gs_uri.py          # dry-run
    python3 backfill_gs_uri.py --apply  # actually update MongoDB
"""

import os
import re
import sys
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
GCS_BUCKET = os.getenv("GCS_BUCKET", "video-marketing-simulation")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

DRY_RUN = "--apply" not in sys.argv


def extract_gs_uri(signed_url: str) -> str | None:
    """Extract gs://bucket/path from a GCS signed URL."""
    if not signed_url:
        return None
    try:
        parsed = urlparse(signed_url)
        # Pattern: https://storage.googleapis.com/{bucket}/{path}?X-Goog-Signature=...
        if "storage.googleapis.com" in parsed.hostname:
            # Path starts with /{bucket}/{blob_path}
            path = unquote(parsed.path).lstrip("/")
            if path.startswith(f"{GCS_BUCKET}/"):
                blob_path = path[len(GCS_BUCKET) + 1:]
                return f"gs://{GCS_BUCKET}/{blob_path}"
        # Pattern: https://{bucket}.storage.googleapis.com/{path}?...
        if parsed.hostname == f"{GCS_BUCKET}.storage.googleapis.com":
            blob_path = unquote(parsed.path).lstrip("/")
            return f"gs://{GCS_BUCKET}/{blob_path}"
    except Exception:
        pass
    return None


def backfill_node_variations():
    """Backfill gs_uri on content_workflow_nodes variations."""
    nodes = db.content_workflow_nodes.find({
        "stage_key": {"$in": ["video_generation", "storyboard"]},
    })
    updated_count = 0
    for node in nodes:
        output = node.get("output_data", {})
        changed = False

        # Video variations
        for var in output.get("variations", []):
            if var.get("gs_uri"):
                continue
            gs_uri = extract_gs_uri(var.get("preview"))
            if gs_uri:
                var["gs_uri"] = gs_uri
                changed = True

        # Storyboard characters & scenes
        for sb in output.get("storyboards", []):
            for item in sb.get("characters", []) + sb.get("scenes", []):
                if item.get("gs_uri"):
                    continue
                gs_uri = extract_gs_uri(item.get("image_url"))
                if gs_uri:
                    item["gs_uri"] = gs_uri
                    changed = True

        if changed:
            updated_count += 1
            node_id = node["_id"]
            stage = node.get("stage_key")
            wf = node.get("workflow_id")
            print(f"  {'[DRY RUN] ' if DRY_RUN else ''}Update node {node_id} (stage={stage}, workflow={wf})")
            if not DRY_RUN:
                db.content_workflow_nodes.update_one(
                    {"_id": node_id},
                    {"$set": {"output_data": output}},
                )

    return updated_count


def backfill_job_videos():
    """Backfill gs_uri on video_generation_jobs videos and sets."""
    jobs = db.video_generation_jobs.find({})
    updated_count = 0
    for job in jobs:
        changed = False

        # Individual scene videos
        for v in job.get("videos", []):
            if v.get("gs_uri"):
                continue
            gs_uri = extract_gs_uri(v.get("video_url"))
            if gs_uri:
                v["gs_uri"] = gs_uri
                changed = True

        # Stitched set URLs
        sets = job.get("sets", {})
        for set_idx, set_info in sets.items():
            if set_info.get("gs_uri"):
                continue
            gs_uri = extract_gs_uri(set_info.get("stitched_url"))
            if gs_uri:
                set_info["gs_uri"] = gs_uri
                changed = True

        if changed:
            updated_count += 1
            job_id = job["_id"]
            task_id = job.get("task_id", "?")
            print(f"  {'[DRY RUN] ' if DRY_RUN else ''}Update job {job_id} (task={task_id})")
            if not DRY_RUN:
                updates = {"videos": job["videos"], "sets": sets}
                db.video_generation_jobs.update_one(
                    {"_id": job_id},
                    {"$set": updates},
                )

    return updated_count


if __name__ == "__main__":
    mode = "DRY RUN" if DRY_RUN else "APPLYING"
    print(f"\n=== Backfill gs_uri ({mode}) ===\n")
    print(f"Database: {MONGODB_DB_NAME}")
    print(f"GCS Bucket: {GCS_BUCKET}\n")

    n1 = backfill_node_variations()
    print(f"\nNodes updated: {n1}")

    n2 = backfill_job_videos()
    print(f"Jobs updated: {n2}")

    print(f"\nTotal documents updated: {n1 + n2}")
    if DRY_RUN:
        print("\nThis was a dry run. Re-run with --apply to actually update MongoDB.")
    print()
