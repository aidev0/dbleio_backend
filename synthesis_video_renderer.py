#!/usr/bin/env python3
"""
Synthesis Video Renderer
Renders synthesis videos based on AI-generated edit plans
"""

import os
import tempfile
import traceback
from datetime import datetime
from typing import Dict, Any, List
from pymongo import MongoClient
from bson import ObjectId
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
from google.cloud import storage
from google.oauth2 import service_account
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Google Cloud Storage Configuration
GCS_BUCKET = os.getenv('GCS_BUCKET', 'video-marketing-simulation')

def get_google_credentials():
    """Load Google Cloud credentials from environment variable"""
    service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if service_account_json:
        try:
            credentials_dict = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            return credentials
        except Exception as e:
            print(f"Error loading service account from env: {e}")
            return None
    return None

# Initialize GCS client
def get_gcs_client():
    """Get or create GCS client"""
    try:
        credentials = get_google_credentials()
        if credentials:
            storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
            return storage_client
        else:
            storage_client = storage.Client()
            return storage_client
    except Exception as e:
        print(f"Warning: Failed to initialize GCS client: {e}")
        return None


def download_video_from_url(url: str, output_path: str) -> bool:
    """Download video from URL (GCS signed URL or HTTP URL)"""
    try:
        print(f"  Downloading video from {url[:100]}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  ✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error downloading video: {e}")
        return False


def upload_video_to_gcs(local_path: str, campaign_id: str, synthesis_video_id: str) -> Dict[str, str]:
    """Upload rendered video to GCS and return URLs"""
    try:
        storage_client = get_gcs_client()
        if not storage_client:
            raise Exception("GCS client not initialized")

        bucket = storage_client.bucket(GCS_BUCKET)

        # Generate blob path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_name = f"campaigns/{campaign_id}/synthesis/{synthesis_video_id}_{timestamp}.mp4"

        blob = bucket.blob(blob_name)

        print(f"  Uploading to GCS: {blob_name}...")
        blob.upload_from_filename(local_path, content_type='video/mp4')

        # Generate signed URL (valid for 7 days)
        credentials = get_google_credentials()
        from datetime import timedelta
        video_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET",
            credentials=credentials
        )

        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

        print(f"  ✓ Uploaded to GCS")

        return {
            "url": video_url,
            "gs_uri": gs_uri
        }
    except Exception as e:
        print(f"  ✗ Error uploading to GCS: {e}")
        raise


def render_synthesis_video(synthesis_video_id: str) -> bool:
    """
    Render a synthesis video based on its timeline

    Args:
        synthesis_video_id: MongoDB ObjectId of the synthesis video

    Returns:
        bool: True if successful, False otherwise
    """

    print("=" * 80)
    print(f"Synthesis Video Renderer - Started")
    print(f"Synthesis Video ID: {synthesis_video_id}")
    print("=" * 80)
    print()

    try:
        # Update status to processing
        db.synthesis_videos.update_one(
            {"_id": ObjectId(synthesis_video_id)},
            {"$set": {
                "status": "processing",
                "processing_started_at": datetime.utcnow()
            }}
        )

        # Get synthesis video record
        synthesis_video = db.synthesis_videos.find_one({"_id": ObjectId(synthesis_video_id)})
        if not synthesis_video:
            raise Exception(f"Synthesis video {synthesis_video_id} not found")

        campaign_id = synthesis_video["campaign_id"]
        timeline = synthesis_video["timeline"]

        print(f"Campaign ID: {campaign_id}")
        print(f"Timeline segments: {len(timeline)}")
        print()

        # Create temp directory for downloaded videos
        temp_dir = tempfile.mkdtemp()
        print(f"Temp directory: {temp_dir}")
        print()

        # Download all source videos
        print("Downloading source videos...")
        source_videos_map = {}  # video_id -> local_path

        for segment in timeline:
            video_id = segment.get("source_video_id")
            if not video_id or video_id in source_videos_map:
                continue

            # Get video from database
            video = db.videos.find_one({"_id": ObjectId(video_id)})
            if not video:
                print(f"  Warning: Video {video_id} not found in database")
                continue

            # Download video
            video_url = video.get("url")
            if not video_url:
                print(f"  Warning: Video {video_id} has no URL")
                continue

            local_path = os.path.join(temp_dir, f"source_{video_id}.mp4")
            if download_video_from_url(video_url, local_path):
                source_videos_map[video_id] = local_path

        print()
        print(f"Downloaded {len(source_videos_map)} source videos")
        print()

        if not source_videos_map:
            raise Exception("No source videos could be downloaded")

        # Process video segments
        print("Processing video segments...")
        clips = []
        total_duration = 0

        for i, segment in enumerate(timeline, 1):
            video_id = segment.get("source_video_id")
            source_start = segment.get("source_start", 0)
            source_end = segment.get("source_end", 0)

            if not video_id or video_id not in source_videos_map:
                print(f"  ✗ Segment {i}: Video {video_id} not available")
                continue

            video_path = source_videos_map[video_id]

            try:
                # Load and cut the video segment
                clip = VideoFileClip(video_path).subclipped(source_start, source_end)
                duration = source_end - source_start
                clips.append(clip)

                print(f"  ✓ Segment {i:2d}: Video {video_id[:8]}... | {source_start:5.1f}s - {source_end:5.1f}s | Duration: {duration:.1f}s | Output: {total_duration:.1f}s - {total_duration + duration:.1f}s")

                total_duration += duration

            except Exception as e:
                print(f"  ✗ Error processing segment {i}: {e}")

        print()
        print(f"Total segments processed: {len(clips)}")
        print(f"Combined duration: {total_duration:.1f} seconds")
        print()

        if not clips:
            raise Exception("No video segments could be loaded")

        # Concatenate all clips
        print("Combining video segments...")
        final_clip = concatenate_videoclips(clips, method="compose")
        print(f"✓ Combined {len(clips)} segments")
        print()

        # Export to temporary file
        output_path = os.path.join(temp_dir, f"synthesis_{synthesis_video_id}.mp4")
        print(f"Rendering video to: {output_path}")

        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            bitrate='5000k',
            logger=None  # Suppress moviepy's verbose logging
        )

        print()
        print("✓ Video rendered successfully")
        print()

        # Upload to GCS
        print("Uploading to Google Cloud Storage...")
        urls = upload_video_to_gcs(output_path, campaign_id, synthesis_video_id)

        # Update database with completed status and video URL
        db.synthesis_videos.update_one(
            {"_id": ObjectId(synthesis_video_id)},
            {"$set": {
                "status": "completed",
                "video_url": urls["url"],
                "gs_uri": urls["gs_uri"],
                "completed_at": datetime.utcnow(),
                "rendered_duration": total_duration,
                "note": f"Successfully rendered {len(clips)} segments into {total_duration:.1f}s video"
            }}
        )

        # Clean up
        print()
        print("Cleaning up...")
        for clip in clips:
            clip.close()
        final_clip.close()

        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        print()
        print("=" * 80)
        print(f"✓ Synthesis video rendered successfully!")
        print(f"Video URL: {urls['url'][:100]}...")
        print("=" * 80)
        print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Error rendering synthesis video: {e}")
        traceback.print_exc()
        print("=" * 80)
        print()

        # Update status to failed
        db.synthesis_videos.update_one(
            {"_id": ObjectId(synthesis_video_id)},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow()
            }}
        )

        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render synthesis video")
    parser.add_argument("synthesis_video_id", help="Synthesis video MongoDB ObjectId")

    args = parser.parse_args()

    success = render_synthesis_video(args.synthesis_video_id)
    exit(0 if success else 1)
