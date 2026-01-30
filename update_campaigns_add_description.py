#!/usr/bin/env python3
"""
Add description field to existing campaigns
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Update all campaigns to have description field if they don't have it
result = db.campaigns.update_many(
    {"description": {"$exists": False}},
    {"$set": {"description": ""}}
)

print(f"Updated {result.modified_count} campaigns to include description field")

# Show all campaigns
campaigns = db.campaigns.find()
print("\nAll campaigns:")
for campaign in campaigns:
    print(f"- {campaign.get('name')}: description = '{campaign.get('description', 'NOT SET')}'")
