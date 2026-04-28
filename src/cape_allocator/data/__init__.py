"""
Data fetching and caching utilities.

Ma et al. (2026) SSRN 6060895 — Component CAPE
Haghani & White (2022) Elm Wealth — Excess Earnings Yield + Merton Rule
Merton (1971) Journal of Economic Theory — Merton Rule formula
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from cape_allocator.models.inputs import (
    EARNINGS_WINDOW_YEARS,
    CapeVariant,
    MarketInputs,
)
from cape_allocator.models.outputs import DataWarning, WarningSeverity

from .fred import fetch_tips_yield
from .shiller import fetch_aggregate_cape
from .yfinance import fetch_component_cape

_LOW_COVERAGE_THRESHOLD = 0.80
_FALLBACK_CODE = "SHILLER_FALLBACK_USED"


def fetch_market_inputs(
    cape_variant: CapeVariant,
) -> tuple[MarketInputs, list[DataWarning]]:
    """
    Fetch current market inputs: CAPE value and TIPS yield.

    Falls back to Shiller aggregate CAPE if constituent coverage < 80%.

    Returns
    -------
    MarketInputs
        Fetched market data.
    list[DataWarning]
        Any warnings from the fetch process.
    """
    warnings: list[DataWarning] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        tips_future = pool.submit(fetch_tips_yield)

        if cape_variant in (
            CapeVariant.COMPONENT_10Y,
            CapeVariant.COMPONENT_5Y,
            CapeVariant.COMPONENT_EWMA,
        ):
            window = EARNINGS_WINDOW_YEARS[cape_variant]
            component_result = fetch_component_cape(window_years=window)
            constituent_coverage = component_result.coverage

            if component_result.coverage >= _LOW_COVERAGE_THRESHOLD:
                cape_value = component_result.cape
            else:
                # Fallback to Shiller aggregate
                cape_value, _ = fetch_aggregate_cape()
                warnings.append(
                    DataWarning(
                        severity=WarningSeverity.INFO,
                        code=_FALLBACK_CODE,
                        message=(
                            "Fell back to Shiller aggregate CAPE due to "
                            f"low constituent coverage ({constituent_coverage:.0%}). "
                            "Aggregate CAPE OOS R² = 46.7% vs 57.5% for Component CAPE "
                            "(Ma et al. 2026, Table 3)."
                        ),
                    )
                )
                cape_variant = CapeVariant.AGGREGATE_10Y
                constituent_coverage = None
        else:
            # Aggregate CAPE
            cape_value, _ = fetch_aggregate_cape()
            constituent_coverage = None

    tips_yield, _ = tips_future.result()

    return MarketInputs(
        cape_value=cape_value,
        tips_yield=tips_yield,
        cape_variant=cape_variant,
        constituent_coverage=constituent_coverage,
        as_of_date=date.today(),
    ), warnings
