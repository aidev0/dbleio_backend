#!/usr/bin/env python3
"""
Authentication middleware for API key validation
"""

import os
from fastapi import HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_HEADER_NAME = "X-API-Key"

# Paths that don't require API key authentication
EXCLUDED_PATHS = [
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/auth/shopify/callback",  # Shopify OAuth callback
]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key for all requests except excluded paths.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip validation for CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip validation for excluded paths
        path = request.url.path
        if any(path == excluded or path.startswith(excluded + "?") for excluded in EXCLUDED_PATHS):
            return await call_next(request)

        # Skip validation if no API key is configured (development mode)
        if not API_KEY:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get(API_KEY_HEADER_NAME)

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing API key. Include 'X-API-Key' header."}
            )

        if api_key != API_KEY:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key"}
            )

        return await call_next(request)


# Keep the dependency version for use in specific routes if needed
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header (dependency version).
    Returns the API key if valid, raises HTTPException if invalid.
    """
    if not API_KEY:
        return ""

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header.",
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key
