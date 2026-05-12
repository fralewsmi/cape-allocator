"""
FastAPI application for cape-allocator.
"""

import logging
import sys
from contextlib import asynccontextmanager
from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from .routers import allocation, health, market, sensitivity

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

try:
    import yfinance as yf

    yf.set_tz_cache_location("/tmp/yfinance")
except Exception:
    pass

cors_origins = getenv("CORS_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cape Allocator API starting up")
    yield
    logger.info("Cape Allocator API shutting down")


app = FastAPI(
    title="Cape Allocator API",
    description="Optimal equity/TIPS allocation using Component CAPE and Merton Rule",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(market.router)
app.include_router(allocation.router)
app.include_router(sensitivity.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


handler = Mangum(app, lifespan="off")
