#!/usr/bin/env python3
"""
Test simulation to debug persona/video loading issues
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGODB_URI)
db = client['dble_db']

def test_simulation_data():
    print("=" * 80)
    print("TESTING SIMULATION DATA")
    print("=" * 80)

    # Get a campaign
    campaigns = list(db['campaigns'].find().limit(1))
    if not campaigns:
        print("❌ No campaigns found!")
        return

    campaign = campaigns[0]
    campaign_id = str(campaign['_id'])
    print(f"\n✓ Found campaign: {campaign.get('name')} (ID: {campaign_id})")

    # Get personas for this campaign
    personas = list(db['personas'].find({'campaign_id': campaign_id}))
    print(f"\n✓ Found {len(personas)} personas for campaign")

    if not personas:
        print("❌ No personas found!")
        return

    # Print first persona structure
    print("\n" + "=" * 80)
    print("PERSONA STRUCTURE (First Persona):")
    print("=" * 80)
    persona = personas[0]
    print(f"ID: {persona.get('_id')}")
    print(f"Name: {persona.get('name')}")
    print(f"Campaign ID: {persona.get('campaign_id')}")
    print(f"\nDemographics keys: {list(persona.get('demographics', {}).keys())}")
    print(f"\nFull demographics:")
    for key, value in persona.get('demographics', {}).items():
        print(f"  {key}: {value}")

    # Check for behavior field
    if 'behavior' in persona:
        print(f"\n✓ Has 'behavior' field: {persona['behavior']}")
    else:
        print("\n❌ No 'behavior' field in persona")

    # Get videos for this campaign
    videos = list(db['videos'].find({'campaign_id': campaign_id}))
    print(f"\n✓ Found {len(videos)} videos for campaign")

    if not videos:
        print("❌ No videos found!")
        return

    # Print first video structure
    print("\n" + "=" * 80)
    print("VIDEO STRUCTURE (First Video):")
    print("=" * 80)
    video = videos[0]
    print(f"ID: {video.get('_id')}")
    print(f"Title: {video.get('title')}")
    print(f"Campaign ID: {video.get('campaign_id')}")
    print(f"\nAnalysis keys: {list(video.get('analysis', {}).keys())}")

    analysis = video.get('analysis', {})
    print(f"\nAnalysis summary: {analysis.get('summary', 'N/A')[:100]}...")
    print(f"Objects: {analysis.get('objects', [])[:5]}")
    print(f"Colors: {analysis.get('colors', [])[:5]}")
    print(f"Scene cuts: {analysis.get('number_of_scene_cut', 0)}")

    # Test formatting logic from simulations.py
    print("\n" + "=" * 80)
    print("TESTING FORMATTING LOGIC:")
    print("=" * 80)

    # Format persona as simulation code does
    formatted_persona = {
        'id': persona.get('id'),
        'name': persona.get('name', 'Unknown'),
        'gender': persona.get('demographics', {}).get('gender', ['Unknown'])[0] if isinstance(persona.get('demographics', {}).get('gender', []), list) else persona.get('demographics', {}).get('gender', 'Unknown'),
        'age_mean': persona.get('demographics', {}).get('age_mean', 0),
        'age_std': persona.get('demographics', {}).get('age_std', 0),
        'num_orders_mean': persona.get('behavior', {}).get('num_orders_mean', 0),  # WRONG!
        'num_orders_std': persona.get('behavior', {}).get('num_orders_std', 0),  # WRONG!
        'revenue_per_customer_mean': persona.get('behavior', {}).get('revenue_per_customer_mean', 0),  # WRONG!
        'revenue_per_customer_std': persona.get('behavior', {}).get('revenue_per_customer_std', 0),  # WRONG!
        'region': persona.get('demographics', {}).get('region', ['Unknown'])[0] if isinstance(persona.get('demographics', {}).get('region', []), list) else persona.get('demographics', {}).get('region', 'Unknown'),
        'weight': persona.get('weight', 0)
    }

    print("\n❌ CURRENT (WRONG) FORMATTING:")
    print(f"  num_orders_mean: {formatted_persona['num_orders_mean']}")
    print(f"  revenue_per_customer_mean: {formatted_persona['revenue_per_customer_mean']}")

    # Correct formatting
    formatted_persona_correct = {
        'id': persona.get('id'),
        'name': persona.get('name', 'Unknown'),
        'gender': persona.get('demographics', {}).get('gender', ['Unknown'])[0] if isinstance(persona.get('demographics', {}).get('gender', []), list) else persona.get('demographics', {}).get('gender', 'Unknown'),
        'age_mean': persona.get('demographics', {}).get('age_mean', 0),
        'age_std': persona.get('demographics', {}).get('age_std', 0),
        'num_orders_mean': persona.get('demographics', {}).get('num_orders_mean', 0),  # CORRECT!
        'num_orders_std': persona.get('demographics', {}).get('num_orders_std', 0),  # CORRECT!
        'revenue_per_customer_mean': persona.get('demographics', {}).get('revenue_per_customer_mean', 0),  # CORRECT!
        'revenue_per_customer_std': persona.get('demographics', {}).get('revenue_per_customer_std', 0),  # CORRECT!
        'region': persona.get('demographics', {}).get('region', ['Unknown'])[0] if isinstance(persona.get('demographics', {}).get('region', []), list) else persona.get('demographics', {}).get('region', 'Unknown'),
        'weight': persona.get('demographics', {}).get('weight', 0)
    }

    print("\n✓ CORRECT FORMATTING:")
    print(f"  num_orders_mean: {formatted_persona_correct['num_orders_mean']}")
    print(f"  revenue_per_customer_mean: {formatted_persona_correct['revenue_per_customer_mean']}")
    print(f"  age_mean: {formatted_persona_correct['age_mean']}")
    print(f"  weight: {formatted_persona_correct['weight']}")

    print("\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_simulation_data()
