#!/usr/bin/env python3
"""
Simple standalone seed script - hardcoded data, no functions
"""

import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Connect
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DB_NAME', 'video_marketing_db')]

# Clear all
print("Clearing database...")
db.campaigns.delete_many({})
db.personas.delete_many({})
db.videos.delete_many({})
print("✓ Cleared")

# Campaign
print("\nInserting campaign...")
campaign = {
    'name': 'Half Price Drapes - Q4 2025',
    'platform': 'Facebook Ads',
    'target_audience': 'New Customers',
    'budget': 85000,
    'details': {
        'description': 'Facebook Ads • New Customer Acquisition • Window Treatments',
        'status': 'active',
        'products': 'Window Treatments',
        'goals': {
            'primary': 'Increase video ad conversions by 25% compared to current baseline',
            'secondary': 'Reduce cost per acquisition (CPA) by 15% through optimized video content',
            'ai_edited': 'Create an optimized video by combining the best-performing moments from all 4 test videos'
        }
    },
    'created_at': datetime.utcnow(),
    'updated_at': datetime.utcnow()
}
campaign_result = db.campaigns.insert_one(campaign)
campaign_id = str(campaign_result.inserted_id)
print(f"✓ Campaign ID: {campaign_id}")

# Personas - all 20 hardcoded
print("\nInserting personas...")
personas = [
    {'campaign_id': campaign_id, 'name': 'Persona 1', 'description': 'Male, 63 years, Northwest', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Male'], 'region': ['Northwest'], 'age_mean': 63.0, 'age_std': 6.98, 'num_orders_mean': 3.57, 'num_orders_std': 2.45, 'revenue_per_customer_mean': 756.54, 'revenue_per_customer_std': 161.46, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 2', 'description': 'Female, 35 years, West', 'demographics': {'age': ['25-34', '35-44'], 'gender': ['Female'], 'region': ['West'], 'age_mean': 35.0, 'age_std': 7.33, 'num_orders_mean': 9.42, 'num_orders_std': 2.27, 'revenue_per_customer_mean': 122.64, 'revenue_per_customer_std': 292.48, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 3', 'description': 'Male, 62 years, South', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Male'], 'region': ['South'], 'age_mean': 62.0, 'age_std': 3.0, 'num_orders_mean': 14.89, 'num_orders_std': 2.04, 'revenue_per_customer_mean': 772.82, 'revenue_per_customer_std': 51.77, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 4', 'description': 'Female, 51 years, Midwest', 'demographics': {'age': ['45-54', '55-64'], 'gender': ['Female'], 'region': ['Midwest'], 'age_mean': 51.0, 'age_std': 6.06, 'num_orders_mean': 2.95, 'num_orders_std': 1.23, 'revenue_per_customer_mean': 503.0, 'revenue_per_customer_std': 164.02, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 5', 'description': 'Male, 68 years, East', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Male'], 'region': ['East'], 'age_mean': 68.0, 'age_std': 5.57, 'num_orders_mean': 9.29, 'num_orders_std': 0.62, 'revenue_per_customer_mean': 768.3, 'revenue_per_customer_std': 92.63, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 6', 'description': 'Female, 28 years, Northeast', 'demographics': {'age': ['18-24', '25-34'], 'gender': ['Female'], 'region': ['Northeast'], 'age_mean': 28.0, 'age_std': 7.71, 'num_orders_mean': 8.89, 'num_orders_std': 1.46, 'revenue_per_customer_mean': 117.56, 'revenue_per_customer_std': 107.72, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 7', 'description': 'Male, 31 years, Northwest', 'demographics': {'age': ['25-34', '35-44'], 'gender': ['Male'], 'region': ['Northwest'], 'age_mean': 31.0, 'age_std': 6.05, 'num_orders_mean': 12.66, 'num_orders_std': 0.93, 'revenue_per_customer_mean': 530.17, 'revenue_per_customer_std': 95.56, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 8', 'description': 'Female, 30 years, West', 'demographics': {'age': ['25-34'], 'gender': ['Female'], 'region': ['West'], 'age_mean': 30.0, 'age_std': 4.04, 'num_orders_mean': 8.95, 'num_orders_std': 0.58, 'revenue_per_customer_mean': 1026.51, 'revenue_per_customer_std': 162.44, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 9', 'description': 'Male, 60 years, South', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Male'], 'region': ['South'], 'age_mean': 60.0, 'age_std': 5.99, 'num_orders_mean': 13.91, 'num_orders_std': 0.72, 'revenue_per_customer_mean': 315.58, 'revenue_per_customer_std': 61.31, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 10', 'description': 'Female, 40 years, Midwest', 'demographics': {'age': ['35-44', '45-54'], 'gender': ['Female'], 'region': ['Midwest'], 'age_mean': 40.0, 'age_std': 6.74, 'num_orders_mean': 8.56, 'num_orders_std': 1.97, 'revenue_per_customer_mean': 1161.78, 'revenue_per_customer_std': 201.76, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 11', 'description': 'Male, 53 years, East', 'demographics': {'age': ['45-54', '55-64'], 'gender': ['Male'], 'region': ['East'], 'age_mean': 53.0, 'age_std': 7.01, 'num_orders_mean': 2.04, 'num_orders_std': 2.97, 'revenue_per_customer_mean': 949.47, 'revenue_per_customer_std': 99.68, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 12', 'description': 'Female, 35 years, Northeast', 'demographics': {'age': ['25-34', '35-44'], 'gender': ['Female'], 'region': ['Northeast'], 'age_mean': 35.0, 'age_std': 3.99, 'num_orders_mean': 10.96, 'num_orders_std': 2.48, 'revenue_per_customer_mean': 766.56, 'revenue_per_customer_std': 281.58, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 13', 'description': 'Male, 52 years, Northwest', 'demographics': {'age': ['45-54', '55-64'], 'gender': ['Male'], 'region': ['Northwest'], 'age_mean': 52.0, 'age_std': 7.32, 'num_orders_mean': 9.73, 'num_orders_std': 1.33, 'revenue_per_customer_mean': 169.91, 'revenue_per_customer_std': 127.75, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 14', 'description': 'Female, 59 years, West', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Female'], 'region': ['West'], 'age_mean': 59.0, 'age_std': 5.96, 'num_orders_mean': 4.85, 'num_orders_std': 1.9, 'revenue_per_customer_mean': 521.22, 'revenue_per_customer_std': 292.93, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 15', 'description': 'Male, 29 years, South', 'demographics': {'age': ['18-24', '25-34'], 'gender': ['Male'], 'region': ['South'], 'age_mean': 29.0, 'age_std': 5.81, 'num_orders_mean': 11.79, 'num_orders_std': 1.73, 'revenue_per_customer_mean': 675.01, 'revenue_per_customer_std': 156.89, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 16', 'description': 'Female, 37 years, Midwest', 'demographics': {'age': ['35-44'], 'gender': ['Female'], 'region': ['Midwest'], 'age_mean': 37.0, 'age_std': 4.01, 'num_orders_mean': 13.54, 'num_orders_std': 1.69, 'revenue_per_customer_mean': 719.6, 'revenue_per_customer_std': 223.88, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 17', 'description': 'Male, 63 years, East', 'demographics': {'age': ['55-64', '65+'], 'gender': ['Male'], 'region': ['East'], 'age_mean': 63.0, 'age_std': 5.05, 'num_orders_mean': 11.58, 'num_orders_std': 1.07, 'revenue_per_customer_mean': 184.68, 'revenue_per_customer_std': 122.44, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 18', 'description': 'Female, 26 years, Northeast', 'demographics': {'age': ['18-24', '25-34'], 'gender': ['Female'], 'region': ['Northeast'], 'age_mean': 26.0, 'age_std': 6.12, 'num_orders_mean': 5.14, 'num_orders_std': 0.76, 'revenue_per_customer_mean': 602.19, 'revenue_per_customer_std': 104.61, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 19', 'description': 'Male, 35 years, Northwest', 'demographics': {'age': ['25-34', '35-44'], 'gender': ['Male'], 'region': ['Northwest'], 'age_mean': 35.0, 'age_std': 4.62, 'num_orders_mean': 2.71, 'num_orders_std': 1.39, 'revenue_per_customer_mean': 1097.51, 'revenue_per_customer_std': 118.03, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
    {'campaign_id': campaign_id, 'name': 'Persona 20', 'description': 'Female, 37 years, West', 'demographics': {'age': ['25-34', '35-44'], 'gender': ['Female'], 'region': ['West'], 'age_mean': 37.0, 'age_std': 7.3, 'num_orders_mean': 1.1, 'num_orders_std': 1.78, 'revenue_per_customer_mean': 559.15, 'revenue_per_customer_std': 105.53, 'weight': 0.05, 'race': [], 'careers': [], 'education': [], 'locations': [], 'country': [], 'zip_codes': [], 'income_level': [], 'household_count': [], 'household_type': [], 'custom_fields': {}}, 'ai_generated': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()},
]

db.personas.insert_many(personas)
print(f"✓ Inserted {len(personas)} personas")

print("\n✓ DONE! Campaign ID:", campaign_id)
client.close()
