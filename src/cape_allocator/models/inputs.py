"""
Input models — investor parameters and market inputs.

All models use Pydantic v2 for runtime validation with strict domain bounds
drawn from the academic literature.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class CapeVariant(StrEnum):
    """
    CAPE earnings-averaging methodologies evaluated in Ma et al. (2026).

    Out-of-sample R² figures are from Table 3, constant-slope approach,
    OOS period 1974–2015.

    COMPONENT_10Y is the recommended default: it achieves the highest
    predictive accuracy and is the paper's primary result.
    """

    COMPONENT_10Y = "component_10y"    # Ma et al. baseline — OOS R² = 57.52%
    COMPONENT_5Y = "component_5y"      # Ma et al. — OOS R² = 55.05%
    COMPONENT_EWMA = "component_ewma"  # Ma et al. — OOS R² = 56.80%
    AGGREGATE_10Y = "aggregate_10y"    # Traditional Shiller — OOS R² = 46.67%


# ── Historical mean CAPEs from Ma et al. (2026), Table 1 (1964–2024) ─────────
HISTORICAL_MEAN_CAPE: dict[CapeVariant, float] = {
    CapeVariant.COMPONENT_10Y:  29.74,
    CapeVariant.COMPONENT_5Y:   24.48,
    CapeVariant.COMPONENT_EWMA: 29.04,
    CapeVariant.AGGREGATE_10Y:  21.65,
}

# Earnings window in years for each variant
EARNINGS_WINDOW_YEARS: dict[CapeVariant, int] = {
    CapeVariant.COMPONENT_10Y:  10,
    CapeVariant.COMPONENT_5Y:   5,
    CapeVariant.COMPONENT_EWMA: 10,
    CapeVariant.AGGREGATE_10Y:  10,
}


class InvestorParams(BaseModel):
    """
    Investor-supplied parameters governing the Merton Rule calculation.

    Defaults follow Haghani & White (2022), who set γ = 2 as representative
    of a slightly risk-tolerant investor, and σ = 18% as the long-run
    equity volatility constant used throughout their historical simulation
    (and adopted by the AllocateSmartly implementation of the strategy).

    Reference:
        Haghani, V. & White, J. (2022). "Man Doth Not Invest By Earnings
        Yield Alone." Elm Wealth.
        https://elmwealth.com/earnings-yield-dynamic-allocation/
    """

    gamma: float = Field(
        default=2.0,
        ge=0.5,
        le=20.0,
        description=(
            "Coefficient of relative risk aversion (CRRA). "
            "γ = 2 is Haghani & White (2022) default; "
            "Ma et al. (2026) Table 8 uses γ = 5."
        ),
    )
    sigma: float = Field(
        default=0.18,
        ge=0.05,
        le=0.60,
        description=(
            "Expected annualised equity volatility (as a decimal). "
            "0.18 (18%) is the constant used by Haghani & White (2022) "
            "and AllocateSmartly's implementation of the strategy."
        ),
    )
    min_equity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum equity allocation (floor), as a decimal.",
    )
    max_equity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.5,
        description=(
            "Maximum equity allocation (cap), as a decimal. "
            "Values above 1.0 imply leverage."
        ),
    )
    cape_variant: CapeVariant = Field(
        default=CapeVariant.COMPONENT_10Y,
        description="CAPE earnings-averaging methodology to use.",
    )

    @model_validator(mode="after")
    def bounds_are_consistent(self) -> InvestorParams:
        if self.min_equity >= self.max_equity:
            raise ValueError(
                f"min_equity ({self.min_equity}) must be strictly less than "
                f"max_equity ({self.max_equity})."
            )
        return self


class MarketInputs(BaseModel):
    """
    Market data inputs required for the allocation calculation.

    Typically populated by the data-fetching layer, but can be constructed
    manually for testing or scripted use.

    cape_value:
        The CAPE ratio for the chosen variant.  Must be positive.
        As of end-2024, the Component 10Y CAPE stood at approximately
        56× (Ma et al., 2026, communicated to Globe and Mail, March 2026),
        versus the historical mean of 29.74×.

    tips_yield:
        The 10-year TIPS real yield as a decimal (e.g. 0.022 for 2.2%).
        Sourced from FRED series DFII10.
        As of early 2026, approximately 2.0–2.2%.

    as_of_date:
        The date for which these market inputs apply.
    """

    cape_value: float = Field(
        gt=0.0,
        description="CAPE ratio (price / cyclically-adjusted earnings). Must be > 0.",
    )
    tips_yield: float = Field(
        ge=-0.10,
        le=0.20,
        description="10-year TIPS real yield as a decimal (e.g. 0.022 = 2.2%).",
    )
    cape_variant: CapeVariant = Field(
        default=CapeVariant.COMPONENT_10Y,
        description="Which CAPE variant this value represents.",
    )
    constituent_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of S&P 500 market cap successfully fetched (0–1). "
            "None if using aggregate data. Below 0.80 triggers fallback."
        ),
    )
    as_of_date: date = Field(
        default_factory=date.today,
        description="Market data as-of date.",
    )
