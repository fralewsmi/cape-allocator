# CAPE Allocator

[![CI](https://github.com/fralewsmi/cape-allocator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fralewsmi/cape-allocator/actions/workflows/ci.yml?query=branch%3Amain)

API backend for [fralewsmi/cape-allocator-ui](https://github.com/fralewsmi/cape-allocator-ui).

I became interested in optimal equity allocation after reading Shiller's Irrational Exuberance and tracking his CAPE ratio during the high valuations of 2025. [This FTAV article](https://www.ft.com/content/84b8a579-8634-47de-a421-a1eb39c8577d) by Toby Nangle pointed me to Ma, Marshall, Nguyen & Visaltanachoti (2026), who proposed the component CAPE as a better predictor of long-run returns. I wanted to test it inside the Merton Rule framework from [Haghani & White (2022)](https://elmwealth.com/earnings-yield-dynamic-allocation/), using the excess earnings yield over TIPS as the equity risk premium.

**Merton Rule:**

$$f^* = \frac{\mu}{\gamma \cdot \sigma^2}$$

where:

- $\mu$ = Excess Earnings Yield = $\frac{1}{\text{CAPE}} - \text{TIPS yield}$
- $\gamma$ = risk aversion
- $\sigma$ = equity volatility

## Installation

### Core library

```bash
# Create and activate a virtual environment
uv venv && source .venv/bin/activate

# Install everything (recommended for development)
uv sync --extra dev

# Or install only what you need
uv sync --extra test      # pytest, pytest-cov, hypothesis
uv sync --extra lint      # ruff
uv sync --extra type      # ty
uv sync --extra api       # fastapi, mangum, uvicorn, httpx

# Copy the environment file and add your FRED API key
cp .env.example .env
```

### API server

```bash
export FRED_API_KEY="your_fred_api_key"
export CORS_ORIGINS="https://your-frontend.com"  # optional, defaults to *

# Run locally
uvicorn api.main:app --reload

# Or with Docker
docker build -t cape-allocator .
docker run -p 8000:8000 -e FRED_API_KEY=... cape-allocator
```

### AWS Lambda deployment

The API deploys to Lambda via the `Mangum` handler in `api/main.py` and the Serverless Framework config in `serverless.yml`. GitHub Actions handles deployment automatically.

Pushes to `main` deploy the Lambda after lint, type check, and tests pass. You can also trigger a manual deploy from the GitHub Actions UI with a custom stage or region.

The stack provisions two Lambda functions:

- **`api`** — the FastAPI handler, invoked by API Gateway on every request
- **`warmer`** — runs at 4am and 4pm UTC to pre-populate the S3 cache, so the `api` function never does a cold data fetch during a user request

An S3 bucket for the shared cache is created automatically (`cape-allocator-cache-<stage>-<account-id>`). Both functions share it via the `CAPE_CACHE_URL` environment variable (`s3://bucket-name`), wired up by the Serverless Framework.

Required GitHub repository secrets:

- `AWS_ROLE_TO_ASSUME`: IAM role ARN trusted by GitHub OIDC for this repository
- `FRED_API_KEY`: FRED API key passed to the Lambda environment

Optional GitHub repository variables:

- `AWS_REGION`: defaults to `ap-southeast-2`
- `SERVERLESS_STAGE`: defaults to `dev`
- `CORS_ORIGINS`: defaults to `https://cape-allocator-ui.lewissmith-fraser.workers.dev`

Use GitHub OIDC rather than static AWS access keys so CI does not store long-lived credentials.

## Usage

### CLI

```bash
cape-allocator # interactive
cape-allocator --gamma 2.0 --sigma 0.18 --momentum-weight 0.5 --cape-variant component_10y
cape-allocator --cape 56.0 --tips 0.022  # manual override, no API needed
```

- `--gamma` has the most impact. `γ = 2` (Haghani & White default) is aggressive; `γ = 5` (Ma et al. calibration) allocates ~30% at the historical mean CAPE.
- `--momentum-weight` controls blending with 12-month S&P 500 momentum (0.0 = pure Merton, 0.5 = equal blend).
- `--cape` and `--tips` together skip live data fetches.
- `--sigma` can usually stay at the default 18%, the long-run historical average.

Run `cape-allocator --help` for all options.

### API

#### Endpoints

- `GET /health` — health check with cache and FRED status
- `GET /api/market-inputs` — current CAPE and TIPS data
- `GET /api/cape-variants` — available CAPE variants
- `POST /api/allocation` — compute allocation with live data
- `POST /api/allocation/manual` — compute allocation with manual inputs
- `GET /api/sensitivity` — stream sensitivity analysis (NDJSON)

#### Example requests

```bash
# Health check
curl http://localhost:8000/health

# Get market inputs
curl "http://localhost:8000/api/market-inputs?cape_variant=component_10y"

# Compute allocation
curl -X POST http://localhost:8000/api/allocation \
  -H "Content-Type: application/json" \
  -d '{"gamma": 2.0, "sigma": 0.18}'

# Manual allocation
curl -X POST http://localhost:8000/api/allocation/manual \
  -H "Content-Type: application/json" \
  -d '{"gamma": 2.0, "sigma": 0.18, "cape_value": 30.0, "tips_yield": 0.02}'
```

### Choosing γ (risk aversion)

A useful heuristic: how would a permanent 50% loss of wealth affect your life?

| γ   | Profile                                                                   |
| --- | ------------------------------------------------------------------------- |
| 1   | Young investor, long horizon, stable income; near-maximally aggressive    |
| 2   | Haghani & White (2022) default; moderate risk tolerance                   |
| 5   | Ma et al. (2026) calibration; pre-retiree or institutionally conservative |
| 10  | Retiree; portfolio is primary income source                               |

`γ` should reflect financial risk aversion, not emotional comfort. A large pension or guaranteed income effectively lowers your financial `γ` even if markets make you nervous. See [Haghani & White (2018)](https://elmwealth.com/measuring-the-fabric-of-felicity/).

### Momentum overlay

The model includes a 12-month momentum overlay following Haghani & White (2022) and Asness et al. (2013):

$$f_\text{blended} = (1 - w) \cdot f_\text{merton} + w \cdot f_\text{momentum}$$

where:

- $f_\text{momentum} = 1.0$ if 12-month S&P 500 momentum is positive, $0.0$ otherwise
- $w$ is the momentum weight (0.0 = pure Merton, 1.0 = pure momentum)

The signal uses the price return from 12 months ago to 1 month ago, excluding the most recent month to avoid short-term reversal.

Use `--momentum-weight 0.5` for equal blending (Asness et al. recommendation). When Merton suggests 0% but momentum is positive, this allocates 50% to equities without arbitrary clamps.

## Data sources

Responses are cached to avoid redundant upstream fetches. The cache backend is selected automatically:

- **Local / Docker**: JSON files under `CAPE_CACHE_URL` (default `~/.cache/cape_allocator`)
- **Lambda**: S3 bucket provisioned by the stack, shared across all function instances (`CAPE_CACHE_URL=s3://bucket-name`)

Sources:

- **FRED** ([API key](https://fred.stlouisfed.org/docs/api/api_key.html) in `.env`): TIPS `DFII10` / `WFII10`, CPI `CPIAUCSL`
- **[Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)**: S&P 500 tickers
- **[Yahoo Finance](https://finance.yahoo.com/)** via [yfinance](https://github.com/ranaroussi/yfinance): prices, market cap, EPS, and monthly S&P 500 prices for momentum (unofficial)
- **[Shiller CAPE spreadsheet](http://www.econ.yale.edu/~shiller/data/ie_data.xls)** (Yale): aggregate CAPE and low-coverage fallback

`--cape` and `--tips` together skip live CAPE/TIPS fetches. Adjust fetch log verbosity with `-v` (verbose) or `-q` (quiet).

## Development

### Optional dependencies

| Group   | Contents |
| ------- | -------- |
| `test`  | pytest, pytest-cov, hypothesis |
| `lint`  | ruff |
| `type`  | ty |
| `api`   | fastapi, mangum, uvicorn, httpx |
| `dev`   | all of the above |

Install any combination: `uv sync --extra test --extra lint`, or `uv sync --extra dev` for everything.

### Type checking

```bash
uv sync --extra type
ty check
```

### Pre-commit hooks

[pre-commit](https://pre-commit.com/) runs ruff, ruff-format, and pytest before each commit.

```bash
pre-commit install        # first time only
pre-commit run --all-files
```

### Linting

```bash
uv sync --extra lint
ruff check .          # check
ruff check . --fix    # auto-fix
ruff format .         # format
```

### Testing

```bash
uv sync --extra test
pytest
pytest --cov=cape_allocator --cov-report=html
```

### Continuous integration

GitHub Actions runs on push to `main`, pull requests to `main`, and manual dispatch. The pipeline runs:

1. Lint (ruff)
2. Type check (ty)
3. Tests (pytest with coverage)
4. Deploy to AWS Lambda (on `main` or manual dispatch)

## References

- Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum Everywhere." _The Journal of Finance_, 68(3), 929–985.
  <https://doi.org/10.1111/jofi.12021>

- Asness, C. S., Ilmanen, A., & Maloney, T. (2017). "Market Timing: Sin a Little — Resolving the Valuation Timing Puzzle." AQR.
  <https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Market-Timing-Sin-a-Little.pdf>

- Haghani, V., & White, J. (2018). "Measuring the Fabric of Felicity." Elm Wealth.
  <https://elmwealth.com/measuring-the-fabric-of-felicity/>

- Haghani, V., & White, J. (2022). "Man Doth Not Invest by Earnings Yield Alone: A Fresh Look at Earnings Yield and Dynamic Asset Allocation." Elm Wealth.
  <https://elmwealth.com/earnings-yield-dynamic-allocation/>

- Li, K., Li, Y., Lyu, C., & Yu, J. (2025). "How to Dominate the Historical Average." _Review of Financial Studies_.
  <https://academic.oup.com/rfs/article/38/10/3086/8010588>

- Ma, Q., Marshall, A., Nguyen, T. H., & Visaltanachoti, N. (2026). "CAPE Ratios and Long-Term Returns."
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6060895>
