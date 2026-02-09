#!/usr/bin/env python3
"""
Migration: Embed organizations into users and add new org fields.

1. Adds brand_name, brand_description, product_description, industry, logo_url to all orgs
2. Reads organization_memberships and embeds full org details into each user's organizations[]
3. Reports results

Usage:
    cd dbleio_backend
    source venv/bin/activate  # or however you activate your env
    python migrate_orgs_to_users.py
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

NEW_ORG_FIELDS = {
    "brand_name": None,
    "brand_description": None,
    "product_description": None,
    "industry": None,
    "logo_url": None,
}


def org_to_embed(org, role, joined_at):
    return {
        "_id": str(org["_id"]),
        "name": org.get("name", ""),
        "slug": org.get("slug", ""),
        "description": org.get("description"),
        "url": org.get("url"),
        "brand_name": org.get("brand_name"),
        "brand_description": org.get("brand_description"),
        "product_description": org.get("product_description"),
        "industry": org.get("industry"),
        "logo_url": org.get("logo_url"),
        "created_by": org.get("created_by"),
        "created_at": org.get("created_at"),
        "updated_at": org.get("updated_at"),
        "role": role,
        "joined_at": joined_at,
    }


def main():
    # Step 1: Add new fields to all existing organizations
    print("=== Step 1: Add new fields to organizations ===")
    orgs = list(db.organizations.find())
    print(f"Found {len(orgs)} organizations")
    for org in orgs:
        update = {}
        for field, default in NEW_ORG_FIELDS.items():
            if field not in org:
                update[field] = default
        if update:
            db.organizations.update_one({"_id": org["_id"]}, {"$set": update})
            print(f"  Updated org '{org.get('name')}' with new fields: {list(update.keys())}")
        else:
            print(f"  Org '{org.get('name')}' already has all fields")

    # Step 2: Initialize organizations array on all users
    print("\n=== Step 2: Initialize users.organizations[] ===")
    users_without = db.users.count_documents({"organizations": {"$exists": False}})
    if users_without > 0:
        db.users.update_many(
            {"organizations": {"$exists": False}},
            {"$set": {"organizations": []}}
        )
        print(f"  Initialized organizations[] on {users_without} users")
    else:
        print("  All users already have organizations field")

    # Step 3: Read memberships and embed into users
    print("\n=== Step 3: Migrate organization_memberships → users.organizations[] ===")
    if "organization_memberships" not in db.list_collection_names():
        print("  No organization_memberships collection found — nothing to migrate")
        return

    memberships = list(db.organization_memberships.find())
    print(f"Found {len(memberships)} memberships to migrate")

    migrated = 0
    skipped = 0
    errors = 0

    for mem in memberships:
        user_id = mem.get("user_id")
        org_id = mem.get("organization_id")
        role = mem.get("role", "member")
        joined_at = mem.get("joined_at")

        # Find the user
        user = db.users.find_one({"workos_user_id": user_id})
        if not user:
            print(f"  SKIP: User {user_id} not found")
            skipped += 1
            continue

        # Check if already embedded
        existing_orgs = user.get("organizations", [])
        if any(o.get("_id") == org_id for o in existing_orgs):
            print(f"  SKIP: User {user.get('email')} already has org {org_id}")
            skipped += 1
            continue

        # Find the org
        try:
            org = db.organizations.find_one({"_id": ObjectId(org_id)})
        except Exception:
            org = None

        if not org:
            print(f"  SKIP: Org {org_id} not found for user {user.get('email')}")
            skipped += 1
            continue

        # Embed
        embed = org_to_embed(org, role, joined_at)
        try:
            db.users.update_one(
                {"_id": user["_id"]},
                {"$push": {"organizations": embed}}
            )
            print(f"  OK: {user.get('email')} ← org '{org.get('name')}' (role={role})")
            migrated += 1
        except Exception as e:
            print(f"  ERROR: {user.get('email')} ← org '{org.get('name')}': {e}")
            errors += 1

    print(f"\n=== Results ===")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")

    # Step 4: Verify
    print("\n=== Step 4: Verification ===")
    users_with_orgs = db.users.count_documents({"organizations.0": {"$exists": True}})
    total_users = db.users.count_documents({})
    print(f"  {users_with_orgs}/{total_users} users have at least one organization")

    print("\n=== Done ===")
    print("The organization_memberships collection is no longer used.")
    print("You can drop it manually when ready:")
    print("  db.organization_memberships.drop()")


if __name__ == "__main__":
    main()
