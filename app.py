#!/usr/bin/env python3
"""
FastAPI application for Video Marketing Simulation
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Import routers
from src.campaigns import router as campaigns_router
from src.personas import router as personas_router
from src.videos import router as videos_router
from src.simulations import router as simulations_router, evaluations_router
from src.insights import router as insights_router
from src.synthesis import router as synthesis_router
from src.users import router as users_router
from src.chat import router as chat_router
from src.tasks import router as tasks_router
from src.integrations import router as integrations_router, auth_router as integrations_auth_router
from src.feature_requests import router as feature_requests_router
from src.plans import router as plans_router
from src.support_chat import router as support_chat_router
from src.contact_forms import router as contact_forms_router
from src.organizations import router as organizations_router
from src.projects import router as projects_router
from src.development_specifications import router as development_specifications_router
from src.development_workflows import router as development_workflows_router
from src.timeline_entries import router as timeline_entries_router
from src.custom_workflows import router as custom_workflows_router
from src.custom_workflow_timeline import router as custom_workflow_timeline_router
from src.brands import router as brands_router
from src.seats import router as seats_router
from src.audiences import router as audiences_router
from src.brand_assets import router as brand_assets_router
from src.content_workflows import router as content_workflows_router
from src.content_timeline import router as content_timeline_router
from src.strategies import router as strategies_router
from src.instagram import router as instagram_router
from src.auth import APIKeyMiddleware

# Initialize FastAPI app
app = FastAPI(title="Video Marketing Simulation API")

# Middleware order matters - they execute in REVERSE order of addition
# Add APIKeyMiddleware first, then CORS, so CORS wraps around auth responses
# This ensures CORS headers are added to all responses including auth errors

# API Key authentication middleware (runs second - after CORS)
app.add_middleware(APIKeyMiddleware)

# CORS configuration (runs first - wraps all responses including auth errors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3333",
        "https://dble.io",
        "https://www.dble.io",
        "https://dbleio-frontend-15e04e0b3c03.herokuapp.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Explicit OPTIONS handler for CORS preflight - handles all paths
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    """Handle CORS preflight requests explicitly"""
    origin = request.headers.get("origin", "")
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3333",
        "https://dble.io",
        "https://www.dble.io",
        "https://dbleio-frontend-15e04e0b3c03.herokuapp.com"
    ]

    if origin in allowed_origins:
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age": "86400",
            }
        )
    return Response(status_code=400)

# Include routers
app.include_router(users_router)
app.include_router(campaigns_router)
app.include_router(personas_router)
app.include_router(videos_router)
app.include_router(simulations_router)
app.include_router(evaluations_router)
app.include_router(insights_router)
app.include_router(synthesis_router)
app.include_router(chat_router)
app.include_router(tasks_router)
app.include_router(integrations_router)
app.include_router(integrations_auth_router)
app.include_router(feature_requests_router)
app.include_router(plans_router)
app.include_router(support_chat_router)
app.include_router(contact_forms_router)
app.include_router(organizations_router)
app.include_router(projects_router)
app.include_router(development_specifications_router)
app.include_router(development_workflows_router)
app.include_router(timeline_entries_router)
app.include_router(custom_workflows_router)
app.include_router(custom_workflow_timeline_router)
app.include_router(brands_router)
app.include_router(seats_router)
app.include_router(audiences_router)
app.include_router(brand_assets_router)
app.include_router(content_workflows_router)
app.include_router(content_timeline_router)
app.include_router(strategies_router)
app.include_router(instagram_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Video Marketing Simulation API", "version": "1.0.0"}

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)
