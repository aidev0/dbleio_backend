#!/usr/bin/env python3
"""
Seed database with original demo data (20 personas with statistical fields)
"""

import os
import json
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in environment variables")

MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Path to frontend data
FRONTEND_PERSONAS_PATH = Path(__file__).parent.parent / 'frontend' / 'public' / 'data' / 'customers' / 'personas.json'

def convert_demo_persona_to_db_format(demo_persona, campaign_id):
    """Convert demo persona format to database format"""
    demographics = demo_persona['demographics']

    # Convert gender to array format
    gender = [demographics['gender']] if demographics.get('gender') else []

    # Convert region to array format
    region = [demographics['region']] if demographics.get('region') else []

    # Age range based on age_mean and age_std (statistical distribution)
    age_mean = demographics.get('age_mean', 30)
    age_std = demographics.get('age_std', 5)

    # Calculate range as mean ± std (covers ~68% of distribution)
    age_min = max(18, age_mean - age_std)
    age_max = age_mean + age_std

    # Map to age brackets
    age_brackets = {
        '18-24': (18, 24),
        '25-34': (25, 34),
        '35-44': (35, 44),
        '45-54': (45, 54),
        '55-64': (55, 64),
        '65+': (65, 120)
    }

    # Find all overlapping brackets
    age = []
    for bracket_name, (bracket_min, bracket_max) in age_brackets.items():
        # Check if the statistical range overlaps with this bracket
        if not (age_max < bracket_min or age_min > bracket_max):
            age.append(bracket_name)

    # If no overlaps found (shouldn't happen), fall back to closest bracket
    if not age:
        if age_mean < 25:
            age = ['18-24']
        elif age_mean < 35:
            age = ['25-34']
        elif age_mean < 45:
            age = ['35-44']
        elif age_mean < 55:
            age = ['45-54']
        elif age_mean < 65:
            age = ['55-64']
        else:
            age = ['65+']

    return {
        'campaign_id': campaign_id,
        'name': demo_persona['name'],
        'description': f"Demo persona: {demographics.get('gender', 'N/A')} from {demographics.get('region', 'N/A')} region, average age {age_mean:.0f}",
        'demographics': {
            # New format - arrays
            'age': age,
            'gender': gender,
            'region': region,
            'locations': [],
            'country': [],
            'zip_codes': [],
            'race': [],
            'careers': [],
            'education': [],
            'income_level': [],
            'household_count': [],
            'household_type': [],
            'custom_fields': {},

            # Original demo format - statistical fields
            'age_mean': demographics.get('age_mean'),
            'age_std': demographics.get('age_std'),
            'num_orders_mean': demographics.get('num_orders_mean'),
            'num_orders_std': demographics.get('num_orders_std'),
            'revenue_per_customer_mean': demographics.get('revenue_per_customer_mean'),
            'revenue_per_customer_std': demographics.get('revenue_per_customer_std'),
            'weight': demographics.get('weight')
        },
        'ai_generated': False,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

def seed_demo_personas():
    """Seed database with demo personas"""
    print("🌱 Seeding demo personas...")

    # Get the first campaign
    campaign = db.campaigns.find_one()
    if not campaign:
        print("❌ No campaign found. Please run seed_database.py first to create a campaign.")
        return

    campaign_id = str(campaign['_id'])
    print(f"📋 Using campaign: {campaign['name']} (ID: {campaign_id})")

    # Load demo personas from JSON
    if not FRONTEND_PERSONAS_PATH.exists():
        print(f"❌ Demo personas file not found: {FRONTEND_PERSONAS_PATH}")
        return

    with open(FRONTEND_PERSONAS_PATH, 'r') as f:
        demo_personas = json.load(f)

    print(f"📊 Found {len(demo_personas)} demo personas")

    # Delete existing personas for this campaign
    deleted = db.personas.delete_many({'campaign_id': campaign_id})
    print(f"🗑️  Deleted {deleted.deleted_count} existing personas")

    # Convert and insert demo personas
    personas_to_insert = []
    for demo_persona in demo_personas:
        db_persona = convert_demo_persona_to_db_format(demo_persona, campaign_id)
        personas_to_insert.append(db_persona)

    if personas_to_insert:
        result = db.personas.insert_many(personas_to_insert)
        print(f"✅ Successfully seeded {len(result.inserted_ids)} demo personas")

        # Verify
        count = db.personas.count_documents({'campaign_id': campaign_id})
        print(f"✓ Total personas in database for this campaign: {count}")

        # Show sample
        sample = db.personas.find_one({'campaign_id': campaign_id})
        if sample:
            print(f"\n📋 Sample persona:")
            print(f"   Name: {sample['name']}")
            print(f"   Age mean: {sample['demographics'].get('age_mean')}")
            print(f"   Gender: {sample['demographics'].get('gender')}")
            print(f"   Region: {sample['demographics'].get('region')}")
            print(f"   Weight: {sample['demographics'].get('weight')}")
    else:
        print("❌ No personas to insert")

if __name__ == '__main__':
    seed_demo_personas()
