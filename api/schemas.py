"""
API-specific request/response schemas.
"""

from typing import Any

from pydantic import BaseModel, Field

from cape_allocator.models.inputs import (
    CapeVariant,
)
from cape_allocator.models.outputs import AllocationResult, DataWarning


class AllocationRequest(BaseModel):
    """Request body for POST /api/allocation."""

    gamma: float = Field(default=2.0, ge=0.5, le=20.0)
    sigma: float = Field(default=0.18, ge=0.05, le=0.60)

    cape_variant: CapeVariant = Field(default=CapeVariant.COMPONENT_10Y)


class ManualAllocationRequest(AllocationRequest):
    """Request body for POST /api/allocation/manual."""

    cape_value: float = Field(gt=0.0)
    tips_yield: float = Field(ge=-0.10, le=0.20)


class AllocationResponse(AllocationResult):
    """Response for allocation endpoints."""

    pass  # AllocationResult already has all fields


class MarketInputsResponse(BaseModel):
    """Response for GET /api/market-inputs."""

    cape_value: float
    tips_yield: float
    cape_variant: CapeVariant
    constituent_coverage: float | None
    as_of_date: str  # ISO format
    warnings: list[DataWarning]


class CapeVariantInfo(BaseModel):
    """Info for a CAPE variant."""

    variant: str
    label: str
    oos_r2: float
    historical_mean: float
    earnings_window_years: int
    description: str


class CapeVariantsResponse(BaseModel):
    """Response for GET /api/cape-variants."""

    variants: list[CapeVariantInfo]


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str  # "healthy" | "degraded"
    cache_age_hours: float | None
    fred_reachable: bool
    as_of: str  # ISO datetime


class ErrorResponse(BaseModel):
    """Structured error response."""

    detail: str
    warnings: list[Any] | None = None
