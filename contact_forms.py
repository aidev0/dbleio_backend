"""
Contact Forms API
Handles contact form submissions from pricing page
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
import os
from pymongo import MongoClient

router = APIRouter(prefix="/api/contact", tags=["contact"])

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client.dble


class ContactFormSubmission(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    job_title: Optional[str] = None
    company: str
    website: Optional[str] = None
    annual_ad_spend: Optional[str] = None
    selected_plan: Optional[str] = None
    message: Optional[str] = None


@router.post("")
async def submit_contact_form(form: ContactFormSubmission):
    """
    Submit a contact form from the pricing page
    """
    try:
        doc = {
            "first_name": form.first_name,
            "last_name": form.last_name,
            "email": form.email,
            "phone": form.phone,
            "job_title": form.job_title,
            "company": form.company,
            "website": form.website,
            "annual_ad_spend": form.annual_ad_spend,
            "selected_plan": form.selected_plan,
            "message": form.message,
            "status": "new",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = db.client_contact_us_forms.insert_one(doc)

        return {
            "success": True,
            "id": str(result.inserted_id),
            "message": "Thank you for your interest! We'll be in touch shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_contact_forms():
    """
    Get all contact form submissions (admin endpoint)
    """
    try:
        forms = list(db.client_contact_us_forms.find().sort("created_at", -1))
        for form in forms:
            form["_id"] = str(form["_id"])
        return forms
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
