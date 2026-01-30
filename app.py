#!/usr/bin/env python3
"""
FastAPI application for Video Marketing Simulation
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Import routers
from campaigns import router as campaigns_router
from personas import router as personas_router
from videos import router as videos_router
from simulations import router as simulations_router, evaluations_router
from insights import router as insights_router
from synthesis import router as synthesis_router
from users import router as users_router
from chat import router as chat_router
from tasks import router as tasks_router
from integrations import router as integrations_router, auth_router as integrations_auth_router
from feature_requests import router as feature_requests_router
from plans import router as plans_router
from support_chat import router as support_chat_router
from contact_forms import router as contact_forms_router
from auth import APIKeyMiddleware

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
