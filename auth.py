#!/usr/bin/env python3
"""
Authentication middleware for API key and JWT validation
Supports both API key (server-to-server) and JWT (browser) authentication
"""

import os
import jwt
import httpx
from fastapi import HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from dotenv import load_dotenv
from functools import lru_cache
from typing import Optional, Dict, Any

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_HEADER_NAME = "X-API-Key"

# WorkOS configuration
WORKOS_CLIENT_ID = os.getenv("WORKOS_CLIENT_ID")
WORKOS_API_KEY = os.getenv("WORKOS_API_KEY")
# JWKS URL - WorkOS uses /sso/jwks/{clientId} for all JWT verification
WORKOS_JWKS_URL = f"https://api.workos.com/sso/jwks/{WORKOS_CLIENT_ID}" if WORKOS_CLIENT_ID else None

# Debug output at startup
print(f"[AUTH CONFIG] WORKOS_CLIENT_ID: {WORKOS_CLIENT_ID}")
print(f"[AUTH CONFIG] WORKOS_JWKS_URL: {WORKOS_JWKS_URL}")

# Paths that don't require authentication
EXCLUDED_PATHS = [
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/auth/shopify/callback",  # Shopify OAuth callback
    "/api/users/auth/callback",  # WorkOS auth callback (needs to work without token)
    "/api/users/auth/refresh",  # Token refresh endpoint (needs to work with expired token)
]

# Cache for JWKS keys
_jwks_cache: Optional[Dict[str, Any]] = None


def get_jwks() -> Dict[str, Any]:
    """Fetch and cache JWKS from WorkOS"""
    global _jwks_cache
    if _jwks_cache is None:
        if not WORKOS_JWKS_URL:
            print("WORKOS_CLIENT_ID not configured - cannot fetch JWKS")
            return {"keys": []}
        try:
            print(f"Fetching JWKS from: {WORKOS_JWKS_URL}")
            response = httpx.get(WORKOS_JWKS_URL, timeout=10.0)
            response.raise_for_status()
            _jwks_cache = response.json()
            print(f"Successfully fetched JWKS with {len(_jwks_cache.get('keys', []))} keys")
        except Exception as e:
            print(f"Failed to fetch JWKS: {e}")
            return {"keys": []}
    return _jwks_cache


def get_signing_key(token: str) -> Optional[str]:
    """Get the signing key for a JWT token from JWKS"""
    try:
        # Decode header to get key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            return None

        jwks = get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                # Convert JWK to PEM format
                from jwt.algorithms import RSAAlgorithm
                return RSAAlgorithm.from_jwk(key)

        # Key not found, try refreshing JWKS cache
        global _jwks_cache
        _jwks_cache = None
        jwks = get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                from jwt.algorithms import RSAAlgorithm
                return RSAAlgorithm.from_jwk(key)

    except Exception as e:
        print(f"Error getting signing key: {e}")

    return None


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a WorkOS JWT token and return the payload
    Returns None if verification fails
    """
    try:
        signing_key = get_signing_key(token)
        if not signing_key:
            print("Could not find signing key for token")
            return None

        # Verify and decode the token
        # WorkOS User Management tokens don't include an audience claim,
        # so we disable audience verification and only verify signature + expiration
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_aud": False,  # WorkOS tokens don't have audience claim
            }
        )

        # Verify issuer matches WorkOS
        issuer = payload.get("iss", "")
        if not issuer.startswith("https://api.workos.com"):
            print(f"Invalid JWT issuer: {issuer}")
            return None

        print(f"[JWT] Successfully verified token for user: {payload.get('sub')}")
        return payload

    except jwt.ExpiredSignatureError:
        print("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid JWT token: {e}")
        return None
    except Exception as e:
        print(f"JWT verification error: {e}")
        return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate authentication via API key or JWT token.
    Accepts either:
    - X-API-Key header (for server-to-server communication)
    - Authorization: Bearer <token> header (for browser requests with JWT)
    """

    async def dispatch(self, request: Request, call_next):
        # Skip validation for CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip validation for excluded paths
        path = request.url.path

        # Check for exact match, path with query string, or path with trailing slash
        for excluded in EXCLUDED_PATHS:
            if path == excluded or path.startswith(excluded + "?") or path.startswith(excluded + "/"):
                return await call_next(request)
            # Also check if the excluded path ends with a pattern that should match
            if excluded.startswith("/api/") and path.startswith(excluded):
                return await call_next(request)

        # Skip validation if no API key is configured (development mode)
        if not API_KEY:
            return await call_next(request)

        # Check for API key first (server-to-server)
        api_key = request.headers.get(API_KEY_HEADER_NAME)
        if api_key:
            if api_key == API_KEY:
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Invalid API key"}
                )

        # Check for JWT token (browser requests)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            payload = verify_jwt_token(token)
            if payload:
                # Store user info in request state for route handlers
                request.state.user = payload
                return await call_next(request)
            else:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token"}
                )

        # No valid authentication provided
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <token>' header."}
        )


# Keep old name for backwards compatibility
APIKeyMiddleware = AuthMiddleware


# Security schemes for OpenAPI docs
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


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


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> Dict[str, Any]:
    """
    Get current user from JWT token (dependency version).
    Returns user payload if valid, raises HTTPException if invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    payload = verify_jwt_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return payload


def get_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get authenticated user from request state (set by middleware).
    Returns None if no user is authenticated or if using API key auth.
    """
    return getattr(request.state, 'user', None)


def get_workos_user_id(request: Request) -> Optional[str]:
    """
    Get the WorkOS user ID from the authenticated user.
    Returns the 'sub' claim from JWT which is the WorkOS user ID.
    Returns None if not authenticated with JWT.
    """
    user = get_user_from_request(request)
    if user:
        return user.get('sub')
    return None


def require_user_id(request: Request) -> str:
    """
    Require and return the WorkOS user ID from the authenticated user.
    Raises HTTPException if not authenticated with JWT.
    Use this for routes that require user identity.
    """
    user_id = get_workos_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User authentication required. Please log in.",
        )
    return user_id
