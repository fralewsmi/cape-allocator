"""
Command-line interface for cape-allocator.

Usage examples
--------------
    # Fully interactive (prompts for all parameters):
    cape-allocator

    # Fully specified via flags:
    cape-allocator --gamma 2.0 --sigma 0.18 --momentum-weight 0.5

    # Mixed: flags provided for some, prompts for the rest:
    cape-allocator --gamma 3.0 --cape-variant component_10y

    # Override CAPE/TIPS manually (skip live data fetch):
    cape-allocator --cape 56.0 --tips 0.022

    # Clear the local cache then run:
    cape-allocator --clear-cache

    # Verbose fetch progress (FRED, Wikipedia, Yahoo, Shiller):
    cape-allocator -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import FloatPrompt, Prompt
from rich.rule import Rule
from rich.table import Table

from cape_allocator.calculations.allocator import (
    compute_allocation,
    fetch_market_inputs_and_allocate,
)
from cape_allocator.models.inputs import CapeVariant, InvestorParams, MarketInputs
from cape_allocator.models.outputs import AllocationResult

console = Console()

_VARIANT_CHOICES = [v.value for v in CapeVariant]

# ── Annotation strings ────────────────────────────────────────────────────────
_GAMMA_HELP = (
    "Risk aversion γ (CRRA).  Default 2.0 per Haghani & White (2022). "
    "Ma et al. (2026) Table 8 uses γ=5."
)
_SIGMA_HELP = (
    "Expected annualised equity volatility σ (decimal).  "
    "Default 0.18 per Haghani & White (2022)."
)
_MOMENTUM_WEIGHT_HELP = (
    "Weight for momentum overlay (decimal).  "
    "0.0 = pure Merton, 0.5 = equal blend (Asness et al. 2013).  "
    "Default 0.0 to preserve current behavior."
)
_VARIANT_HELP = (
    "CAPE variant.  component_10y is the Ma et al. (2026) baseline "
    "(OOS R²=57.5 percent).  aggregate_10y is traditional Shiller "
    "(OOS R²=46.7 percent)."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cape-allocator",
        description=(
            "Component CAPE + Merton Rule portfolio allocator.\n\n"
            "Computes the optimal equity/TIPS split for a CRRA investor "
            "using the Component CAPE methodology of Ma et al. (2026) and "
            "the Excess Earnings Yield framework of Haghani & White (2022)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gamma", type=float, default=None, help=_GAMMA_HELP)
    parser.add_argument("--sigma", type=float, default=None, help=_SIGMA_HELP)
    parser.add_argument(
        "--momentum-weight",
        type=float,
        default=None,
        help=_MOMENTUM_WEIGHT_HELP,
    )
    parser.add_argument(
        "--cape-variant",
        choices=_VARIANT_CHOICES,
        default=None,
        help=_VARIANT_HELP,
    )
    parser.add_argument(
        "--cape",
        type=float,
        default=None,
        help="Override: supply CAPE value directly (skips yfinance fetch).",
    )
    parser.add_argument(
        "--tips",
        type=float,
        default=None,
        help="Override: supply TIPS yield directly as decimal (skips FRED fetch).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached market data before running.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="More detail in fetch progress (-vv for debug).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress fetch progress logging.",
    )
    return parser


def _configure_cli_logging(*, verbose: int, quiet: bool) -> None:
    """Attach a Rich log handler to ``cape_allocator`` loggers (CLI only)."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    log = logging.getLogger("cape_allocator")
    log.handlers.clear()
    log.setLevel(level)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    handler.setLevel(level)
    log.addHandler(handler)
    log.propagate = False


def _prompt_if_none(
    value: float | None, label: str, default: float, min_val: float, max_val: float
) -> float:
    if value is not None:
        return value
    console.print(f"  [dim]{label}[/dim]")
    raw = FloatPrompt.ask("  Enter value", default=default, console=console)
    if not (min_val <= raw <= max_val):
        console.print(
            f"  [red]Value {raw} out of range [{min_val}, {max_val}]. "
            f"Using default {default}.[/red]"
        )
        return default
    return raw


def _prompt_variant_if_none(variant_str: str | None) -> CapeVariant:
    if variant_str is not None:
        return CapeVariant(variant_str)
    console.print()
    console.print("  [bold]CAPE variant[/bold]")
    for i, v in enumerate(_VARIANT_CHOICES, 1):
        oos = {
            "component_10y": "57.5%",
            "component_5y": "55.1%",
            "component_ewma": "56.8%",
            "aggregate_10y": "46.7%",
        }.get(v, "")
        marker = " [green](recommended)[/green]" if v == "component_10y" else ""
        console.print(f"  [{i}] {v}  OOS R²={oos}{marker}")
    choice = Prompt.ask(
        "  Select variant [1-4]",
        choices=["1", "2", "3", "4"],
        default="1",
        console=console,
    )
    return CapeVariant(_VARIANT_CHOICES[int(choice) - 1])


def _render_result(result: AllocationResult) -> None:
    """Render the allocation result as a Rich table."""

    # ── Warnings panel ────────────────────────────────────────────────────
    if result.warnings:
        for w in result.warnings:
            colour = {"INFO": "blue", "WARN": "yellow", "ERROR": "red"}.get(
                w.severity.value, "white"
            )
            console.print(f"  [{colour}][{w.severity.value}][/{colour}] {w.message}")
        console.print()

    # ── Main table ────────────────────────────────────────────────────────
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("", style="dim", width=34)
    table.add_column("Value", justify="right", min_width=14)
    table.add_column("Notes", style="dim", max_width=46)

    # Section: Market Inputs
    table.add_row("[bold]Market inputs[/bold]", "", "")
    table.add_row(
        "  CAPE ratio",
        f"{result.cape_value:.1f}×",
        f"{result.cape_variant.value}  (hist. mean {result.historical_mean_cape:.1f}×)",
    )
    table.add_row(
        "  CAPE vs historical mean",
        f"{result.cape_vs_mean_pct:+.1f}%",
        "Ma et al. (2026) Table 1, 1964–2024",
    )
    table.add_row(
        "  10yr TIPS real yield",
        f"{result.tips_yield:.3%}",
        "FRED DFII10",
    )
    if result.constituent_coverage is not None:
        table.add_row(
            "  Constituent coverage",
            f"{result.constituent_coverage:.0%}",
            "Fraction of S&P 500 market cap fetched",
        )

    table.add_row("", "", "")

    # Section: Derived Signals
    table.add_row("[bold]Derived signals[/bold]", "", "")
    table.add_row(
        "  Earnings yield (1/CAPE)",
        f"{result.earnings_yield:.3%}",
        "Real equity return estimate",
    )
    eey_colour = "green" if result.excess_earnings_yield >= 0 else "red"
    table.add_row(
        "  Excess earnings yield μ",
        f"[{eey_colour}]{result.excess_earnings_yield:+.3%}[/{eey_colour}]",
        "EY − TIPS yield  (Haghani & White, 2022)",
    )
    constrained_note = (
        " [dim](constrained)[/dim]" if result.allocation_is_constrained else ""
    )
    table.add_row(
        "  Merton share (unconstrained)",
        f"{result.merton_share_unconstrained:.1%}",
        f"μ / (γ·σ²)  γ={result.gamma}  σ={result.sigma:.0%}  (Merton, 1971)",
    )
    table.add_row(
        "  Momentum signal",
        f"{result.momentum_signal:+.1%}",
        "12-month S&P 500 return (t-12 to t-1)  (Haghani & White, 2022)",
    )
    table.add_row(
        "  Momentum allocation",
        f"{result.f_momentum:.0%}",
        "Binary: 100% if momentum > 0, else 0%  (Asness et al., 2013)",
    )

    table.add_row("", "", "")

    # Section: Allocation
    table.add_row("[bold]Optimal allocation[/bold]", "", "")
    eq_colour = "green" if result.equity_allocation >= 0.4 else "yellow"
    table.add_row(
        "  Equities",
        (
            f"[bold {eq_colour}]{result.equity_allocation:.1%}"
            f"[/bold {eq_colour}]{constrained_note}"
        ),
        "S&P 500",
    )
    table.add_row(
        "  TIPS",
        f"[bold]{result.tips_allocation:.1%}[/bold]",
        "10-year inflation-protected bonds",
    )
    table.add_row(
        "  Certainty equiv. return",
        f"{result.cer:.3%}",
        "f·μ − (γ/2)·(f·σ)²  (Ma et al. 2026, eq. 17)",
    )

    table.add_row("", "", "")

    # Section: Investor Params
    table.add_row("[bold]Investor parameters[/bold]", "", "")
    table.add_row("  Risk aversion γ", f"{result.gamma}", "CRRA coefficient")
    table.add_row("  Equity volatility σ", f"{result.sigma:.0%}", "")
    table.add_row(
        "  Momentum weight",
        f"{result.momentum_weight:.0%}",
        "Weight for momentum overlay (0.0 = pure Merton)",
    )

    # Footer
    table.add_row("", "", "")
    table.add_row(
        "  As of",
        str(result.as_of_date),
        "",
    )

    console.print(
        Panel(
            table,
            title="[bold]Cape Allocator — Portfolio Allocation[/bold]",
            subtitle=("Ma et al. (2026) · Haghani & White (2022) · Merton (1971)"),
            padding=(1, 2),
        )
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_cli_logging(verbose=args.verbose, quiet=args.quiet)

    if args.clear_cache:
        from cape_allocator.data.cache import cache_clear

        cache_clear()
        console.print("[green]Cache cleared.[/green]")

    console.print()
    console.print(Rule("[bold]Cape Allocator[/bold]"))
    console.print()
    console.print(
        "  Computes the optimal equity/TIPS allocation using the\n"
        "  [bold]Component CAPE[/bold] (Ma et al. 2026) and the\n"
        "  [bold]Merton Rule[/bold] (Haghani & White 2022).\n"
        "  Missing flags will be prompted interactively.\n"
    )

    # ── Collect investor parameters (flags → prompts fallback) ────────────
    console.print(Rule("Investor parameters", style="dim"))
    console.print()

    gamma = _prompt_if_none(
        args.gamma,
        "Risk aversion γ  (Haghani & White default: 2.0; Ma et al. use 5.0)",
        default=2.0,
        min_val=0.5,
        max_val=20.0,
    )
    sigma = _prompt_if_none(
        args.sigma,
        "Equity volatility σ  (Haghani & White default: 0.18)",
        default=0.18,
        min_val=0.05,
        max_val=0.60,
    )
    momentum_weight = _prompt_if_none(
        args.momentum_weight,
        "Momentum overlay weight (e.g. 0.5 for equal blend with Merton)",
        default=0.0,
        min_val=0.0,
        max_val=1.0,
    )
    variant = _prompt_variant_if_none(args.cape_variant)

    try:
        investor = InvestorParams(
            gamma=gamma,
            sigma=sigma,
            momentum_weight=momentum_weight,
            cape_variant=variant,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Invalid investor parameters: {exc}[/red]")
        sys.exit(1)

    console.print()
    console.print(Rule("Fetching market data", style="dim"))
    console.print()

    # ── Market data: manual override or live fetch ─────────────────────────
    if args.cape is not None and args.tips is not None:
        console.print(
            f"  [dim]Using manual overrides: CAPE={args.cape:.1f}×  "
            f"TIPS={args.tips:.3%}[/dim]"
        )
        market = MarketInputs(
            cape_value=args.cape,
            tips_yield=args.tips,
            cape_variant=variant,
            constituent_coverage=None,
            as_of_date=date.today(),
        )
        result = compute_allocation(investor, market)
    else:
        if args.cape is not None or args.tips is not None:
            console.print(
                "  [yellow]Both --cape and --tips must be supplied together "
                "to skip live fetching.  Fetching live data instead.[/yellow]"
            )
        if not args.quiet:
            console.print(
                "  [dim]Live fetch progress is logged below "
                "(-v for more detail, -q to hide).[/dim]"
            )
            console.print()
        try:
            result = fetch_market_inputs_and_allocate(investor)
        except OSError as exc:
            console.print(f"\n[red]Configuration error: {exc}[/red]")
            sys.exit(1)
        except RuntimeError as exc:
            console.print(f"\n[red]Data fetch error: {exc}[/red]")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[red]Unexpected error: {exc}[/red]")
            sys.exit(1)

    console.print()
    console.print(Rule("Result", style="dim"))
    console.print()
    _render_result(result)
    console.print()

    if result.has_errors():
        sys.exit(1)


if __name__ == "__main__":
    main()
