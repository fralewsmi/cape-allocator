"""
Cache warmer — pre-populates the shared cache for all data sources.

Fetches all data sources in dependency order:
  1. CPI (needed by component CAPE)
  2. TIPS yield (FRED)
  3. Shiller aggregate CAPE (Yale XLS)
  4. S&P 500 tickers (Wikipedia)
  5. Component CAPE — 10-year window  (500 yfinance calls, slow)
  6. Component CAPE — 5-year window
  7. S&P 500 monthly prices (momentum signal)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

_STEPS = [
    ("cpi", "cape_allocator.data.cpi", "fetch_cpi_index", ()),
    ("tips_yield", "cape_allocator.data.fred", "fetch_tips_yield", ()),
    ("shiller_cape", "cape_allocator.data.shiller", "fetch_aggregate_cape", ()),
    ("sp500_tickers", "cape_allocator.data.yfinance", "fetch_sp500_tickers", ()),
    (
        "component_cape_10y",
        "cape_allocator.data.yfinance",
        "fetch_component_cape",
        (10,),
    ),
    ("component_cape_5y", "cape_allocator.data.yfinance", "fetch_component_cape", (5,)),
    (
        "sp500_monthly_prices",
        "cape_allocator.data.yfinance",
        "fetch_sp500_monthly_prices",
        (),
    ),
]


def warm_all() -> dict[str, str]:
    """
    Fetch and cache all data sources. Returns a status dict per step.
    Continues past individual failures so one bad source doesn't block others.
    """
    import importlib

    results: dict[str, str] = {}
    for name, module_path, fn_name, args in _STEPS:
        try:
            module = importlib.import_module(module_path)
            getattr(module, fn_name)(*args)
            results[name] = "ok"
            logger.info("Warmer: %s — done", name)
        except Exception as exc:  # noqa: BLE001
            results[name] = f"error: {exc}"
            logger.exception("Warmer: %s — failed", name)

    logger.info("Warmer complete: %s", results)
    return results


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return warm_all()


if __name__ == "__main__":
    results = warm_all()
    failed = [k for k, v in results.items() if v != "ok"]
    sys.exit(1 if failed else 0)
