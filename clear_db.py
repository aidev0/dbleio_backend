#!/usr/bin/env python3
"""
Script to clear all data from the MongoDB database
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Get all collections
collections = db.list_collection_names()
print(f"Found collections: {collections}")

if not collections:
    print("No collections found in database")
else:
    # Drop all collections
    for collection_name in collections:
        db[collection_name].drop()
        print(f"Dropped collection: {collection_name}")

    print("\nAll data cleared from database!")
