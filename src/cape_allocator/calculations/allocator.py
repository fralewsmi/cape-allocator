"""
Top-level allocation orchestrator.

Chains data fetching → CAPE calculation → Merton Rule → AllocationResult.
Collects all warnings along the way rather than raising exceptions on
data quality issues.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from cape_allocator.models.inputs import CapeVariant, InvestorParams, MarketInputs
from cape_allocator.models.outputs import (
    AllocationResult,
    DataWarning,
    WarningSeverity,
)

from .cape import (
    cape_percentile_vs_history,
    compute_earnings_yield,
)
from .merton import (
    apply_allocation_bounds,
    compute_cer,
    compute_excess_earnings_yield,
    compute_merton_share,
)

_LOW_COVERAGE_THRESHOLD = 0.80
_HIGH_CAPE_DEVIATION_WARN_PCT = 50.0  # Warn if CAPE > 150% of historical mean
_NEGATIVE_EEY_CODE = "NEGATIVE_EXCESS_EARNINGS_YIELD"
_LOW_COVERAGE_CODE = "LOW_CONSTITUENT_COVERAGE"
_FALLBACK_CODE = "SHILLER_FALLBACK_USED"
_CONSTRAINED_CODE = "ALLOCATION_CONSTRAINED"
_HIGH_CAPE_CODE = "CAPE_SIGNIFICANTLY_ABOVE_MEAN"

logger = logging.getLogger(__name__)


def compute_allocation(
    investor: InvestorParams,
    market: MarketInputs,
) -> AllocationResult:
    """
    Compute the optimal equity/TIPS allocation from investor and market inputs.

    This is the library's primary entry point.  It is a pure orchestration
    function — all I/O has already occurred by the time this is called.

    Parameters
    ----------
    investor : InvestorParams
        Investor risk preferences and allocation bounds.
    market : MarketInputs
        Pre-fetched market data (CAPE value, TIPS yield, coverage).

    Returns
    -------
    AllocationResult
        Full result including intermediate signals and any warnings.
    """
    warnings: list[DataWarning] = []

    # ── Step 1: Earnings yield  (EY = 1/CAPE) ─────────────────────────────
    ey = compute_earnings_yield(market.cape_value)

    # ── Step 2: Excess earnings yield  (μ = EY − TIPS) ────────────────────
    eey = compute_excess_earnings_yield(ey, market.tips_yield)

    if eey < 0:
        warnings.append(
            DataWarning(
                severity=WarningSeverity.WARN,
                code=_NEGATIVE_EEY_CODE,
                message=(
                    f"Excess earnings yield is negative ({eey:.4f}). "
                    "TIPS currently offer a higher expected real return than equities. "
                    "The Merton Rule will call for the minimum equity allocation."
                ),
            )
        )

    # ── Step 3: CAPE context warning ──────────────────────────────────────
    pct_above_mean = cape_percentile_vs_history(market.cape_value, market.cape_variant)
    if pct_above_mean > _HIGH_CAPE_DEVIATION_WARN_PCT:
        warnings.append(
            DataWarning(
                severity=WarningSeverity.INFO,
                code=_HIGH_CAPE_CODE,
                message=(
                    f"{market.cape_variant.value} CAPE ({market.cape_value:.1f}x) is "
                    f"{pct_above_mean:.0f}% above its historical mean "
                    f"(Ma et al. 2026, Table 1)."
                ),
            )
        )

    # ── Step 4: Coverage warnings ─────────────────────────────────────────
    if market.constituent_coverage is not None:
        if market.constituent_coverage < _LOW_COVERAGE_THRESHOLD:
            warnings.append(
                DataWarning(
                    severity=WarningSeverity.WARN,
                    code=_LOW_COVERAGE_CODE,
                    message=(
                        f"Constituent coverage was {market.constituent_coverage:.0%}, "
                        f"below the {_LOW_COVERAGE_THRESHOLD:.0%} threshold."
                    ),
                )
            )
        if (
            market.cape_variant == CapeVariant.AGGREGATE_10Y
            and market.constituent_coverage is not None
        ):
            warnings.append(
                DataWarning(
                    severity=WarningSeverity.INFO,
                    code=_FALLBACK_CODE,
                    message=(
                        "Fell back to Shiller aggregate CAPE due to "
                        "low constituent coverage. "
                        "Aggregate CAPE OOS R² = 46.7% vs 57.5% for Component CAPE "
                        "(Ma et al. 2026, Table 3)."
                    ),
                )
            )

    # ── Step 5: Merton Rule ───────────────────────────────────────────────
    # f* = μ / (γ · σ²)   — Merton (1971)
    merton_raw = compute_merton_share(eey, investor.gamma, investor.sigma)

    # ── Step 6: Apply investor bounds ─────────────────────────────────────
    f_star = apply_allocation_bounds(
        merton_raw, investor.min_equity, investor.max_equity
    )

    if f_star != merton_raw:
        bound = "minimum" if merton_raw < investor.min_equity else "maximum"
        warnings.append(
            DataWarning(
                severity=WarningSeverity.INFO,
                code=_CONSTRAINED_CODE,
                message=(
                    f"Unconstrained Merton share ({merton_raw:.1%}) was clamped to "
                    f"the {bound} equity allocation ({f_star:.1%})."
                ),
            )
        )

    # ── Step 7: CER ───────────────────────────────────────────────────────
    # CER = f·μ − (γ/2)·(f·σ)²   — Ma et al. (2026) eq. (17)
    cer = compute_cer(f_star, eey, investor.gamma, investor.sigma)

    return AllocationResult(
        # Inputs echoed
        cape_value=market.cape_value,
        cape_variant=market.cape_variant,
        tips_yield=market.tips_yield,
        gamma=investor.gamma,
        sigma=investor.sigma,
        min_equity=investor.min_equity,
        max_equity=investor.max_equity,
        as_of_date=market.as_of_date,
        constituent_coverage=market.constituent_coverage,
        # Signals
        earnings_yield=ey,
        excess_earnings_yield=eey,
        merton_share_unconstrained=merton_raw,
        equity_allocation=f_star,
        tips_allocation=1.0 - f_star,
        cer=cer,
        warnings=warnings,
    )


def fetch_market_inputs_and_allocate(investor: InvestorParams) -> AllocationResult:
    """
    Convenience function that fetches live market data then computes allocation.

    Applies the coverage-based fallback logic:
    1. Attempt to fetch Component CAPE from yfinance constituents.
    2. If coverage < 80%, fall back to Shiller aggregate CAPE and attach warning.
    3. Fetch TIPS yield from FRED (DFII10).

    Parameters
    ----------
    investor : InvestorParams
        Investor risk preferences and chosen CAPE variant.

    Returns
    -------
    AllocationResult
        Full result with live market data.
    """
    # Import here to keep calculations/ free of I/O at module level
    from cape_allocator.data.fred import fetch_tips_yield
    from cape_allocator.data.shiller import fetch_aggregate_cape
    from cape_allocator.data.yfinance import fetch_component_cape
    from cape_allocator.models.inputs import EARNINGS_WINDOW_YEARS

    variant = investor.cape_variant
    warnings_pre: list[DataWarning] = []
    constituent_coverage: float | None = None

    logger.info(
        "Market data: fetching 10-year TIPS (FRED) in parallel with CAPE inputs…"
    )

    # TIPS is independent of CAPE work; overlap the FRED call with yfinance / Shiller.
    with ThreadPoolExecutor(max_workers=2) as pool:
        tips_future = pool.submit(fetch_tips_yield)
        try:
            if variant in (
                CapeVariant.COMPONENT_10Y,
                CapeVariant.COMPONENT_5Y,
                CapeVariant.COMPONENT_EWMA,
            ):
                window = EARNINGS_WINDOW_YEARS[variant]
                logger.info(
                    "Market data: component CAPE (%s-year window) via Yahoo Finance "
                    "constituents + FRED CPI + Wikipedia ticker list…",
                    window,
                )
                component_result = fetch_component_cape(window_years=window)
                constituent_coverage = component_result.coverage

                if component_result.coverage >= _LOW_COVERAGE_THRESHOLD:
                    cape_value = component_result.cape
                else:
                    # Fallback to Shiller aggregate
                    warnings_pre.append(
                        DataWarning(
                            severity=WarningSeverity.WARN,
                            code=_LOW_COVERAGE_CODE,
                            message=(
                                f"Component CAPE coverage was "
                                f"{component_result.coverage:.0%} "
                                f"(below {_LOW_COVERAGE_THRESHOLD:.0%} threshold). "
                                "Falling back to Shiller aggregate CAPE."
                            ),
                        )
                    )
                    warnings_pre.append(
                        DataWarning(
                            severity=WarningSeverity.INFO,
                            code=_FALLBACK_CODE,
                            message=(
                                "Using Shiller aggregate CAPE (AGGREGATE_10Y) as "
                                "fallback. Aggregate CAPE OOS R² = 46.7% vs 57.5% for "
                                "Component CAPE (Ma et al. 2026, Table 3)."
                            ),
                        )
                    )
                    logger.info(
                        "Market data: low coverage — fetching Shiller aggregate CAPE "
                        "(Yale ie_data.xls)…"
                    )
                    agg_cape, _ = fetch_aggregate_cape()
                    cape_value = agg_cape
                    variant = CapeVariant.AGGREGATE_10Y
            else:
                # AGGREGATE_10Y requested directly
                logger.info("Market data: Shiller aggregate CAPE (Yale ie_data.xls)…")
                agg_cape, _ = fetch_aggregate_cape()
                cape_value = agg_cape
        finally:
            tips_yield, _ = tips_future.result()

    logger.info("Market data: fetch phase complete (CAPE + TIPS).")

    market = MarketInputs(
        cape_value=cape_value,
        tips_yield=tips_yield,
        cape_variant=variant,
        constituent_coverage=constituent_coverage,
        as_of_date=date.today(),
    )

    result = compute_allocation(investor, market)

    # Prepend any pre-computation warnings
    result.warnings[:0] = warnings_pre
    return result
