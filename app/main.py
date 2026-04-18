from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.client import close_client, init_client
from app.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_client()
    yield
    await close_client()


def create_app() -> FastAPI:
    app = FastAPI(
        title="perplexity-proxy",
        description="OpenAI-compatible proxy for Perplexity AI",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
