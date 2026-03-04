"""
Middleware for the Deep Retina Grade API.

Includes:
- Structured JSON logging with request tracing
- Rate limiting per IP address
- Security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection,
  Referrer-Policy, Strict-Transport-Security, Content-Security-Policy)

Author: Deep Retina Grade Project
Date: February 2026
"""

import asyncio
import os
import time
import uuid
import logging
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# =============================================================================
# Structured Logging Middleware
# =============================================================================

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request/response with structured JSON data.
    
    Includes: method, path, status, latency, request_id.
    """
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = logging.getLogger("deep_retina_grade.api")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Attach request_id to request state for downstream use
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"request_id={request_id} method={request.method} "
                f"path={request.url.path} status=500 "
                f"latency_ms={latency_ms:.1f} error={str(e)}"
            )
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        self.logger.info(
            f"request_id={request_id} method={request.method} "
            f"path={request.url.path} status={response.status_code} "
            f"latency_ms={latency_ms:.1f}"
        )
        
        # Add request ID header
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"
        
        return response


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter using sliding window.
    
    Limits requests per IP address. Returns 429 when exceeded.
    
    Args:
        app: FastAPI application
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
    """
    
    def __init__(
        self,
        app: FastAPI,
        max_requests: int = 60,
        window_seconds: int = 60,
        trust_proxy: bool | None = None
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()  # protects creation/deletion of per-IP locks
        # Trust X-Forwarded-For only when explicitly enabled
        if trust_proxy is not None:
            self.trust_proxy = trust_proxy
        else:
            self.trust_proxy = os.getenv("TRUST_X_FORWARDED", "false").lower() == "true"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        if self.trust_proxy:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]
        
        # Prune empty keys to prevent unbounded memory growth
        if not self.requests[client_ip]:
            del self.requests[client_ip]
            return False
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return True
        
        self.requests[client_ip].append(now)
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        # Acquire (or create) the per-IP lock
        async with self._locks_guard:
            if client_ip not in self._locks:
                self._locks[client_ip] = asyncio.Lock()
            ip_lock = self._locks[client_ip]
        
        async with ip_lock:
            if self._is_rate_limited(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please try again later.",
                        "retry_after_seconds": self.window_seconds
                    },
                    headers={
                        "Retry-After": str(self.window_seconds),
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Window": f"{self.window_seconds}s"
                    }
                )
        
        # Prune the per-IP lock when the IP has no remaining request records.
        # Safe to do outside `async with ip_lock` — the lock is no longer held.
        if client_ip not in self.requests:
            async with self._locks_guard:
                # Re-check: another coroutine may have re-created the entry
                if client_ip not in self.requests and client_ip in self._locks:
                    lock = self._locks[client_ip]
                    if not lock.locked():
                        del self._locks[client_ip]
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.max_requests - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Window"] = f"{self.window_seconds}s"
        
        return response


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses.
    
    Follows OWASP recommendations for API security.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Prevent MIME-type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection (legacy, but doesn't hurt)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS — enforce HTTPS (1 year)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        
        # Content-Security-Policy — restrict resource loading
        # FastAPI's /docs (Swagger UI) and /redoc load scripts/styles from CDNs
        # and use inline scripts, so they need a relaxed policy.
        _docs_paths = ("/docs", "/redoc", "/openapi.json")
        if request.url.path in _docs_paths:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "worker-src 'self' blob:"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self'"
            )
        
        # Cache control for API responses (no caching medical data)
        if request.url.path.startswith("/predict") or request.url.path.startswith("/explain"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        
        return response
