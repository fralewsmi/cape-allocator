"""
Allocation endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from cape_allocator.calculations.allocator import (
    compute_allocation,
    fetch_market_inputs_and_allocate,
)
from cape_allocator.models.inputs import InvestorParams, MarketInputs

from ..schemas import AllocationRequest, AllocationResponse, ManualAllocationRequest

router = APIRouter(prefix="/api", tags=["allocation"])


@router.post("/allocation", response_model=AllocationResponse)
async def post_allocation(request: AllocationRequest) -> AllocationResponse:
    """Compute allocation using live market data."""
    investor = InvestorParams(
        gamma=request.gamma,
        sigma=request.sigma,
        cape_variant=request.cape_variant,
    )
    result = fetch_market_inputs_and_allocate(investor)
    if result.has_errors():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result.model_dump()
        )
    return AllocationResponse(**result.model_dump())


@router.post("/allocation/manual", response_model=AllocationResponse)
async def post_allocation_manual(
    request: ManualAllocationRequest,
) -> AllocationResponse:
    """Compute allocation using manual market data."""
    investor = InvestorParams(
        gamma=request.gamma,
        sigma=request.sigma,
        cape_variant=request.cape_variant,
    )
    market = MarketInputs(
        cape_value=request.cape_value,
        tips_yield=request.tips_yield,
        cape_variant=request.cape_variant,
    )
    result = compute_allocation(investor, market)
    if result.has_errors():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=result.model_dump()
        )
    return AllocationResponse(**result.model_dump())
