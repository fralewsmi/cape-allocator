"""
Data fetching and caching utilities.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

logger = logging.getLogger(__name__)

_LOW_COVERAGE_THRESHOLD = 0.80
_EPS_EXCLUSION_WARN_THRESHOLD = 0.10  # Warn if >10% of EPS years excluded
_COMPONENT_CAPE_TIMEOUT_SECONDS = 20  # Budget within the Lambda's 30s limit
_FALLBACK_CODE = "SHILLER_FALLBACK_USED"
_CAPE_UNDERSTATED_CODE = "CAPE_POTENTIALLY_UNDERSTATED"
_CAPE_TIMEOUT_CODE = "COMPONENT_CAPE_TIMEOUT"


def fetch_market_inputs(
    cape_variant: CapeVariant,
) -> tuple[MarketInputs, list[DataWarning]]:
    """
    Fetch current market inputs: CAPE value and TIPS yield.

    Falls back to Shiller aggregate CAPE if:
    - constituent coverage < 80%, or
    - the component CAPE fetch exceeds _COMPONENT_CAPE_TIMEOUT_SECONDS
      (guards against stale-cache cold fetches timing out the Lambda).

    Returns
    -------
    MarketInputs
        Fetched market data.
    list[DataWarning]
        Any warnings from the fetch process.
    """
    warnings: list[DataWarning] = []
    eps_exclusion_rate: float | None = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        tips_future = pool.submit(fetch_tips_yield)

        if cape_variant in (
            CapeVariant.COMPONENT_10Y,
            CapeVariant.COMPONENT_5Y,
            CapeVariant.COMPONENT_EWMA,
        ):
            window = EARNINGS_WINDOW_YEARS[cape_variant]
            component_future = pool.submit(fetch_component_cape, window_years=window)

            try:
                component_result = component_future.result(
                    timeout=_COMPONENT_CAPE_TIMEOUT_SECONDS
                )
            except FuturesTimeoutError:
                # The live fetch is too slow (cache miss on a cold Lambda).
                # Cancel the in-flight work and fall back immediately so we
                # can still return a useful response within the Lambda timeout.
                component_future.cancel()
                logger.warning(
                    "Component CAPE fetch exceeded %ss timeout — falling back to "
                    "Shiller aggregate CAPE. Cache is likely stale; warmer should "
                    "repopulate shortly.",
                    _COMPONENT_CAPE_TIMEOUT_SECONDS,
                )
                cape_value, _ = fetch_aggregate_cape()
                warnings.append(
                    DataWarning(
                        severity=WarningSeverity.WARN,
                        code=_CAPE_TIMEOUT_CODE,
                        message=(
                            f"Component CAPE fetch exceeded "
                            f"{_COMPONENT_CAPE_TIMEOUT_SECONDS}s and was cancelled."
                            " Fell back to Shiller aggregate CAPE. The cache warmer"
                            " will repopulate component CAPE data shortly."
                        ),
                    )
                )
                cape_variant = CapeVariant.AGGREGATE_10Y
                constituent_coverage = None
                tips_yield, _ = tips_future.result()
                return MarketInputs(
                    cape_value=cape_value,
                    tips_yield=tips_yield,
                    cape_variant=cape_variant,
                    constituent_coverage=None,
                    eps_exclusion_rate=None,
                    as_of_date=date.today(),
                ), warnings

            constituent_coverage = component_result.coverage
            eps_exclusion_rate = component_result.eps_exclusion_rate

            # Check for EPS data quality issues
            if component_result.eps_exclusion_rate > _EPS_EXCLUSION_WARN_THRESHOLD:
                warnings.append(
                    DataWarning(
                        severity=WarningSeverity.WARN,
                        code=_CAPE_UNDERSTATED_CODE,
                        message=(
                            f"{component_result.eps_exclusion_rate:.0%} of EPS years "
                            "were excluded due to non-positive values. "
                            "Component CAPE may be understated due to data quality "
                            "limitations (yfinance earnings history is patchy)."
                        ),
                    )
                )

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
                eps_exclusion_rate = None
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
        eps_exclusion_rate=eps_exclusion_rate,
        as_of_date=date.today(),
    ), warnings
