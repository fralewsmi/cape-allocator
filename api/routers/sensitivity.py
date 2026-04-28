"""
Sensitivity analysis endpoint.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from cape_allocator.calculations.cape import compute_earnings_yield
from cape_allocator.calculations.merton import (
    compute_cer,
    compute_excess_earnings_yield,
    compute_merton_share,
)

router = APIRouter(prefix="/api", tags=["sensitivity"])


@router.get("/sensitivity")
async def get_sensitivity(
    gamma_min: int = Query(1, ge=1, le=10),
    gamma_max: int = Query(10, ge=1, le=10),
    cape_min: float = Query(5.0, gt=0),
    cape_max: float = Query(80.0, gt=0),
    cape_step: float = Query(0.5, gt=0),
    tips_yield: float = Query(0.017, ge=-0.1, le=0.2),
    sigma: float = Query(0.18, ge=0.05, le=0.6),
) -> StreamingResponse:
    """Stream sensitivity analysis as NDJSON."""
    import json

    async def generate() -> Any:
        cape_range = [
            cape_min + i * cape_step
            for i in range(int((cape_max - cape_min) / cape_step) + 1)
        ]
        for gamma in range(gamma_min, gamma_max + 1):
            for cape in cape_range:
                ey = compute_earnings_yield(cape)
                eey = compute_excess_earnings_yield(ey, tips_yield)
                merton_raw = compute_merton_share(eey, gamma, sigma)
                equity_allocation = merton_raw
                cer = compute_cer(equity_allocation, eey, gamma, sigma)
                row = {
                    "gamma": gamma,
                    "cape": cape,
                    "equity_allocation": equity_allocation,
                    "cer": cer,
                }
                yield json.dumps(row) + "\n"
            await asyncio.sleep(0)  # Yield control

    return StreamingResponse(generate(), media_type="application/x-ndjson")
