"""
FastAPI application for cape-allocator.
"""

from contextlib import asynccontextmanager
from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from .routers import allocation, health, market, sensitivity

# Load environment variables
load_dotenv()

# CORS origins
cors_origins = getenv("CORS_ORIGINS", "*").split(",")


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


# Create FastAPI app
app = FastAPI(
    title="Cape Allocator API",
    description="Optimal equity/TIPS allocation using Component CAPE and Merton Rule",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(market.router)
app.include_router(allocation.router)
app.include_router(sensitivity.router)

# Mangum handler for Lambda
handler = Mangum(app, lifespan="off")
