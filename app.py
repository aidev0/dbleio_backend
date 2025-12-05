#!/usr/bin/env python3
"""
FastAPI application for Video Marketing Simulation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
        "https://dble.io",
        "https://www.dble.io",
        "https://dbleio-frontend-15e04e0b3c03.herokuapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Video Marketing Simulation API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
