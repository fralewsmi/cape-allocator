# Component CAPE + Merton Rule Portfolio Allocator

I became interested in optimal equity allocation after reading Shiller's Irrational Exuberance and tracking his CAPE ratio during the high equity valuations in 2025 during the AI bubble and Trump's tariffs.

(This FTAV article)<https://www.ft.com/content/84b8a579-8634-47de-a421-a1eb39c8577d> pointed me to Ma, Marshall, Nguyen & Visaltanachoti (2026), who proposed the component CAPE as a new model that provides a higher level of accuracy for returns prediction.

I thought it would be fun to test this out using the Merton Rule framework proposed by Haghani & White (2022), Using the excess yield over the TIPS rate to establish the equity risk premium.

**Merton Rule:**

$$f^* = \frac{\mu}{\gamma \cdot \sigma^2}$$

where:

- $\mu$ = Excess Earnings Yield = $\frac{1}{\text{CAPE}} - \text{TIPS yield}$
- $\gamma$ = risk aversion
- $\sigma$ = equity volatility

## Installation

```bash
# Create and activate virtual environment (or use venv, conda, etc.)
uv venv && source .venv/bin/activate

# Install dependencies (core only for basic usage)
uv pip install -e "."

# OR install with development tools
uv pip install -e ".[dev]"

# Copy environment file
cp .env.example .env   # add your FRED API key
```

## Usage

```bash
cape-allocator         # interactive
cape-allocator --gamma 2.0 --sigma 0.18 --cape-variant component_10y
cape-allocator --cape 56.0 --tips 0.022  # manual override, no API needed
```

## References

This project cites the following sources for the CAPE and Merton Rule ideas used in the allocator.

- Haghani, V., & White, J. (2022). "Man Doth Not Invest by Earnings Yield Alone: A Fresh Look at Earnings Yield and Dynamic Asset Allocation." Elm Wealth.
  - https://elmwealth.com/earnings-yield-dynamic-allocation/

- Li, K., Li, Y., Lyu, C., & Yu, J. (2025). "How to Dominate the Historical Average." _Review of Financial Studies_.
  - https://academic.oup.com/rfs/article/38/10/3086/8010588

- Ma, Q., Marshall, A., Nguyen, T. H., & Visaltanachoti, N. (2026). Component CAPE research.
  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6060895
