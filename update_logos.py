#!/usr/bin/env python3
"""Update organization logo URLs."""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

LOGOS = {
    "dble": "/logo.png",
    "Half Price Drapes": "https://www.halfpricedrapes.com/cdn/shop/files/HPD_logo4-03-croped.png?v=1766765776&width=600",
    "Y Combinator": "https://www.ycombinator.com/arc/arc-logo.png",
}

for org_name, logo_url in LOGOS.items():
    org = db.organizations.find_one({"name": org_name})
    if not org:
        print(f"NOT FOUND: {org_name}")
        continue

    now = datetime.utcnow()
    db.organizations.update_one({"_id": org["_id"]}, {"$set": {"logo_url": logo_url, "updated_at": now}})

    org_id = str(org["_id"])
    db.users.update_many(
        {"organizations._id": org_id},
        {"$set": {"organizations.$[elem].logo_url": logo_url, "organizations.$[elem].updated_at": now}},
        array_filters=[{"elem._id": org_id}],
    )
    print(f"OK: {org_name} → {logo_url}")

print("Done")
