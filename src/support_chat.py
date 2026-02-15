"""
Support Chat API
Simple chat endpoint for customer support
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from pymongo import MongoClient

router = APIRouter(prefix="/api/chat", tags=["chat"])

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]


class ChatMessage(BaseModel):
    role: str
    content: str


class SupportChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class SupportChatResponse(BaseModel):
    response: str


# Simple FAQ responses
FAQ_RESPONSES = {
    "pricing": "We have 4 plans: Platform ($3,000/mo), Starter ($6,000/mo), Team ($10,000/mo), and Enterprise (custom). All plans include full platform access. Visit our pricing page for details.",
    "price": "We have 4 plans: Platform ($3,000/mo), Starter ($6,000/mo), Team ($10,000/mo), and Enterprise (custom). All plans include full platform access. Visit our pricing page for details.",
    "cost": "We have 4 plans: Platform ($3,000/mo), Starter ($6,000/mo), Team ($10,000/mo), and Enterprise (custom). All plans include full platform access. Visit our pricing page for details.",
    "plan": "We have 4 plans: Platform ($3,000/mo), Starter ($6,000/mo), Team ($10,000/mo), and Enterprise (custom). All plans include full platform access. Visit our pricing page for details.",
    "how": "dble builds AI marketing automation systems. You tell us what to automate, we build and deploy the system, and it runs 24/7 generating ads, testing creative, and optimizing spend.",
    "work": "dble builds AI marketing automation systems. You tell us what to automate, we build and deploy the system, and it runs 24/7 generating ads, testing creative, and optimizing spend.",
    "feature": "Our platform includes AI Video Ad Generation, AI Image Ad Generation, Pre-Launch Creative Testing, Live Campaign Optimization, and Reinforcement Learning. All features are available on every plan.",
    "integration": "We integrate with Shopify, Meta (Facebook/Instagram), TikTok, Google Ads, Amazon, and various analytics platforms.",
    "support": "All plans include onboarding, integration support, 24/7 monitoring, and access to our team via Slack, chat, and calls.",
    "contact": "You can reach us at support@dble.io or submit a request through the platform.",
    "demo": "To schedule a demo, please submit a request through the platform or email us at support@dble.io.",
    "trial": "We don't offer free trials, but we're happy to walk you through the platform. Submit a request to get started.",
    "hello": "Hello! How can I help you today?",
    "hi": "Hi! How can I help you today?",
    "hey": "Hey! How can I help you today?",
    "thanks": "You're welcome! Is there anything else I can help with?",
    "thank": "You're welcome! Is there anything else I can help with?",
}

DEFAULT_RESPONSE = "Thanks for your message. Our team will get back to you soon. For immediate help, email support@dble.io."


@router.post("/support", response_model=SupportChatResponse)
async def support_chat(request: SupportChatRequest):
    """
    Handle support chat messages
    """
    try:
        message_lower = request.message.lower()

        # Find matching FAQ response
        response = DEFAULT_RESPONSE
        for keyword, faq_response in FAQ_RESPONSES.items():
            if keyword in message_lower:
                response = faq_response
                break

        # Log the chat message
        db.support_chats.insert_one({
            "message": request.message,
            "response": response,
            "history": [msg.dict() for msg in request.history] if request.history else [],
            "created_at": datetime.utcnow()
        })

        return SupportChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
