"""
Market data and CAPE variants endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from cape_allocator.data import fetch_market_inputs
from cape_allocator.models.inputs import (
    EARNINGS_WINDOW_YEARS,
    HISTORICAL_MEAN_CAPE,
    OOS_R2,
    CapeVariant,
)

from ..schemas import CapeVariantInfo, CapeVariantsResponse, MarketInputsResponse

router = APIRouter(prefix="/api", tags=["market"])


@router.get("/market-inputs", response_model=MarketInputsResponse)
async def get_market_inputs(
    cape_variant: Annotated[
        CapeVariant, Query(description="CAPE")
    ] = CapeVariant.COMPONENT_10Y,
) -> MarketInputsResponse:
    """Fetch current live market data."""
    market_inputs, warnings = fetch_market_inputs(cape_variant)
    return MarketInputsResponse(
        cape_value=market_inputs.cape_value,
        tips_yield=market_inputs.tips_yield,
        cape_variant=market_inputs.cape_variant,
        constituent_coverage=market_inputs.constituent_coverage,
        as_of_date=market_inputs.as_of_date.isoformat(),
        warnings=warnings,
    )


@router.get("/cape-variants", response_model=CapeVariantsResponse)
async def get_cape_variants() -> CapeVariantsResponse:
    """Get metadata about all available CAPE variants."""
    variants = []
    for variant in CapeVariant:
        label = variant.value.replace("_", " ").title()
        description = {
            CapeVariant.COMPONENT_10Y: "Component CAPE 10-year earnings (recommended)",
            CapeVariant.COMPONENT_5Y: "Component CAPE 5-year earnings",
            CapeVariant.COMPONENT_EWMA: "Component CAPE exponentially weighted",
            CapeVariant.AGGREGATE_10Y: "Traditional Shiller aggregate CAPE",
        }.get(variant, "")
        variants.append(
            CapeVariantInfo(
                variant=variant.value,
                label=label,
                oos_r2=OOS_R2[variant],
                historical_mean=HISTORICAL_MEAN_CAPE[variant],
                earnings_window_years=EARNINGS_WINDOW_YEARS[variant],
                description=description,
            )
        )
    return CapeVariantsResponse(variants=variants)
