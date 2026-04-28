"""
Health check endpoint.
"""

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from cape_allocator.data.cache import get_cache_age_hours
from cape_allocator.data.fred import check_fred_connectivity

from ..dependencies import CacheDir, CacheTtlHours, FredApiKey
from ..schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check(
    fred_api_key: FredApiKey,
    cache_dir: CacheDir,
    cache_ttl_hours: CacheTtlHours,
) -> HealthResponse:
    """Check API health, cache freshness, and FRED connectivity."""
    try:
        # Check cache age
        cache_age = get_cache_age_hours(cache_dir)
        cache_fresh = cache_age is None or cache_age < cache_ttl_hours

        # Check FRED connectivity (with timeout)
        fred_reachable = await asyncio.wait_for(
            check_fred_connectivity(fred_api_key), timeout=5.0
        )

        # Determine status
        if fred_reachable and cache_fresh:
            status = "healthy"
        else:
            status = "degraded"

        return HealthResponse(
            status=status,
            cache_age_hours=cache_age,
            fred_reachable=fred_reachable,
            as_of=datetime.now(UTC).isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Health check failed: {str(e)}"
        ) from e
