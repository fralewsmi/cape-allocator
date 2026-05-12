"""
Shared FastAPI dependencies.
"""

from os import getenv
from typing import Annotated

from fastapi import Depends, HTTPException, status


def get_fred_api_key() -> str:
    """Dependency to get FRED API key from environment."""
    key = getenv("FRED_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FRED_API_KEY environment variable is not set",
        )
    return key


def get_cache_ttl_hours() -> float:
    """Get cache TTL hours from environment, default to 24."""
    try:
        return float(getenv("CAPE_CACHE_TTL_HOURS", "24"))
    except ValueError:
        return 24.0


FredApiKey = Annotated[str, Depends(get_fred_api_key)]
CacheTtlHours = Annotated[float, Depends(get_cache_ttl_hours)]
