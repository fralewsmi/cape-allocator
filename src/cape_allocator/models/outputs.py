"""
Output models — allocation result and warning system.

The warnings list allows the library to surface data quality issues
without raising exceptions, so callers can inspect and decide.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from cape_allocator.models.inputs import HISTORICAL_MEAN_CAPE, CapeVariant


class WarningSeverity(StrEnum):
    INFO = "INFO"  # Contextual note, no action required
    WARN = "WARN"  # Data quality issue; result still produced
    ERROR = "ERROR"  # Significant data problem; result may be unreliable


class DataWarning(BaseModel):
    """A single warning attached to an AllocationResult."""

    severity: WarningSeverity
    code: str  # Short machine-readable identifier, e.g. "LOW_COVERAGE"
    message: str  # Human-readable description


class AllocationResult(BaseModel):
    """
    Full output of a single allocation computation.

    All intermediate signals are preserved so callers can inspect,
    log, or display the full reasoning chain — not just the final number.

    Formula references:
        Earnings yield:         EY = 1 / CAPE
            (Campbell & Shiller, 1988)
        Excess earnings yield:  μ = EY − TIPS_yield
            (Haghani & White, 2022)
        Merton share (raw):     f* = μ / (γ · σ²)
            (Merton, 1971, eq. for optimal risky asset weight)
        Certainty equiv. return: CER = f·μ − (γ/2)·(f·σ)²
            (Ma et al., 2026, eq. 17; Campbell & Thompson, 2008)
    """

    # ── Inputs (echoed for traceability) ─────────────────────────────────────
    cape_value: float
    cape_variant: CapeVariant
    tips_yield: float
    gamma: float
    sigma: float
    min_equity: float
    max_equity: float
    as_of_date: date
    constituent_coverage: float | None

    # ── Derived signals ───────────────────────────────────────────────────────
    earnings_yield: float = Field(
        description="1 / CAPE — real equity return estimate (Campbell & Shiller, 1988)."
    )
    excess_earnings_yield: float = Field(
        description=(
            "EY minus TIPS real yield — equity risk premium over TIPS "
            "(Haghani & White, 2022)."
        )
    )
    merton_share_unconstrained: float = Field(
        description="f* = μ / (γ·σ²) before applying allocation bounds (Merton, 1971)."
    )
    equity_allocation: float = Field(
        description=(
            "Constrained optimal equity allocation, "
            "clamped to [min_equity, max_equity]."
        )
    )
    tips_allocation: float = Field(
        description="1 − equity_allocation (residual allocated to TIPS)."
    )
    cer: float = Field(
        description=(
            "Certainty Equivalent Return = f·μ − (γ/2)·(f·σ)². "
            "Risk-adjusted return the investor expects from this allocation "
            "(Ma et al., 2026, eq. 17)."
        )
    )

    # ── Context ───────────────────────────────────────────────────────────────
    warnings: list[DataWarning] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def historical_mean_cape(self) -> float:
        """Historical mean CAPE for this variant from Ma et al. (2026) Table 1."""
        return HISTORICAL_MEAN_CAPE[self.cape_variant]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cape_vs_mean_pct(self) -> float:
        """How far current CAPE deviates from its historical mean, as a percentage."""
        mean = self.historical_mean_cape
        return (self.cape_value - mean) / mean * 100.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allocation_is_constrained(self) -> bool:
        """True if the Merton share was clamped by the investor's bounds."""
        return self.equity_allocation != self.merton_share_unconstrained

    def has_errors(self) -> bool:
        return any(w.severity == WarningSeverity.ERROR for w in self.warnings)
