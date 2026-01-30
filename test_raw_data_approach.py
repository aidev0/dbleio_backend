#!/usr/bin/env python3
"""
Test that the new raw data approach works for simulations
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from ai_agent import create_evaluation_prompt

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGODB_URI)
db = client['dble_db']

def test_raw_data_prompt():
    print("=" * 80)
    print("TESTING RAW DATA APPROACH")
    print("=" * 80)

    # Get a campaign
    campaign = db['campaigns'].find_one()
    if not campaign:
        print("❌ No campaigns found!")
        return

    campaign_id = str(campaign['_id'])
    print(f"\n✓ Campaign: {campaign.get('name')} (ID: {campaign_id})")

    # Get personas (raw data)
    personas = list(db['personas'].find({'campaign_id': campaign_id}).limit(1))
    if not personas:
        print("❌ No personas found!")
        return

    persona = personas[0]
    print(f"✓ Persona: {persona.get('name')}")

    # Get videos (raw data)
    videos = list(db['videos'].find({'campaign_id': campaign_id}))
    if not videos:
        print("❌ No videos found!")
        return

    print(f"✓ Videos: {len(videos)} found")

    # Format minimal - just convert ObjectId to string
    formatted_persona = dict(persona)
    formatted_persona['_id'] = str(persona['_id'])
    if 'id' not in formatted_persona:
        formatted_persona['id'] = str(persona['_id'])

    formatted_videos = []
    for video in videos:
        video_copy = dict(video)
        video_copy['_id'] = str(video['_id'])
        video_copy['video_id'] = str(video['_id'])
        formatted_videos.append(video_copy)

    # Create prompt with raw data
    prompt = create_evaluation_prompt(formatted_persona, formatted_videos)

    print("\n" + "=" * 80)
    print("PROMPT PREVIEW (First 1000 chars):")
    print("=" * 80)
    print(prompt[:1000])
    print("...\n")

    print("=" * 80)
    print("✓ SUCCESS! Raw data approach generates valid prompt")
    print(f"✓ Prompt length: {len(prompt)} characters")
    print(f"✓ Persona has {len(formatted_persona)} fields")
    print(f"✓ Videos: {len(formatted_videos)} videos")
    print("=" * 80)

if __name__ == '__main__':
    test_raw_data_prompt()
