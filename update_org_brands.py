#!/usr/bin/env python3
"""
Update organization brand details and assign YC org to yc@dble.io.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]


ORG_BRANDS = {
    "dble": {
        "brand_name": "dble",
        "brand_description": "dble is an AI-powered platform that helps businesses test, simulate, and optimize their digital workflows. From video marketing simulations to AI-driven development pipelines, dble enables teams to validate ideas before committing resources.",
        "product_description": "AI video marketing simulation platform for testing video campaigns with persona-based audience simulations before ad spend. AI-powered developer workflow automation with spec-to-deploy pipelines, code review, and approval gates.",
        "industry": "AI / SaaS / Marketing Technology",
        "logo_url": "https://dble.io/logo.png",
    },
    "Half Price Drapes": {
        "brand_name": "Half Price Drapes",
        "brand_description": "Half Price Drapes is a family-owned business founded in 2005 in the San Francisco Bay Area, now a leading national e-commerce specialty company. For 20 years they have crafted beautiful, high-quality drapes with exceptional attention to detail, combining timeless craftsmanship with accessible pricing. 4.8/5 rating from thousands of reviews.",
        "product_description": "Premium ready-made curtains and custom draperies across 10,000+ fabrics, textures, and colors starting at $15/panel. Product lines include silk, linen, taffeta, velvet, cotton satin, Italian cotton silk, sheer curtains, blackout curtains, faux linen curtains, roman shades, tie-up shades, and metal hardware.",
        "industry": "Home Decor / E-Commerce / Window Coverings",
        "logo_url": None,
    },
    "Y Combinator": {
        "brand_name": "Y Combinator",
        "brand_description": "Y Combinator (YC) is the world's leading startup accelerator and venture capital firm, founded in 2005 by Paul Graham, Jessica Livingston, Robert Tappan Morris, and Trevor Blackwell. YC has launched 5,000+ companies with a combined valuation exceeding $600 billion. Notable alumni include Airbnb, Stripe, Coinbase, DoorDash, Dropbox, Instacart, and Reddit.",
        "product_description": "Three-month accelerator program run four times a year providing $500K seed funding ($125K for 7% equity + $375K uncapped SAFE), intensive mentorship, office hours with partners, Demo Day investor presentations, Bookface alumni network of 11,000+ founders, and access to top-tier investors and follow-on funding.",
        "industry": "Venture Capital / Startup Accelerator",
        "logo_url": None,
    },
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
    # Step 1: Update org brand details
    print("=== Step 1: Update organization brand details ===")
    for org_name, brand_data in ORG_BRANDS.items():
        org = db.organizations.find_one({"name": org_name})
        if not org:
            print(f"  NOT FOUND: '{org_name}'")
            continue

        now = datetime.utcnow()
        brand_data_with_ts = {**brand_data, "updated_at": now}
        db.organizations.update_one({"_id": org["_id"]}, {"$set": brand_data_with_ts})
        print(f"  UPDATED: '{org_name}' — brand_name={brand_data['brand_name']}, industry={brand_data['industry']}")

        # Sync to all users who have this org embedded
        org_id = str(org["_id"])
        updated_org = db.organizations.find_one({"_id": org["_id"]})
        set_fields = {}
        for field in ["name", "slug", "description", "url", "brand_name", "brand_description", "product_description", "industry", "logo_url", "updated_at"]:
            set_fields[f"organizations.$[elem].{field}"] = updated_org.get(field)
        result = db.users.update_many(
            {"organizations._id": org_id},
            {"$set": set_fields},
            array_filters=[{"elem._id": org_id}],
        )
        print(f"    Synced to {result.modified_count} user(s)")

    # Step 2: Assign YC org to yc@dble.io
    print("\n=== Step 2: Assign Y Combinator org to yc@dble.io ===")
    yc_user = db.users.find_one({"email": "yc@dble.io"})
    if not yc_user:
        print("  User yc@dble.io not found!")
    else:
        yc_org = db.organizations.find_one({"name": "Y Combinator"})
        if not yc_org:
            print("  Y Combinator org not found!")
        else:
            org_id = str(yc_org["_id"])
            existing_orgs = yc_user.get("organizations", [])
            if any(o.get("_id") == org_id for o in existing_orgs):
                print(f"  SKIP: yc@dble.io already has Y Combinator org")
            else:
                now = datetime.utcnow()
                embed = org_to_embed(yc_org, role="owner", joined_at=now)
                db.users.update_one(
                    {"_id": yc_user["_id"]},
                    {"$push": {"organizations": embed}},
                )
                print(f"  OK: yc@dble.io ← Y Combinator (role=owner)")

    # Step 3: Verify
    print("\n=== Verification ===")
    for org_name in ORG_BRANDS:
        org = db.organizations.find_one({"name": org_name})
        if org:
            print(f"  {org_name}: brand_name={org.get('brand_name')}, industry={org.get('industry')}")

    yc_user = db.users.find_one({"email": "yc@dble.io"})
    if yc_user:
        org_names = [o.get("name") for o in yc_user.get("organizations", [])]
        print(f"  yc@dble.io orgs: {org_names}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
