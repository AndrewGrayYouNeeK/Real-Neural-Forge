"""FastAPI application factory and entry-point."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from neural_forge.api.routes import predict, health

app = FastAPI(
    title="Neural Forge API",
    description="REST API for Neural Forge model inference.",
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/predict", tags=["inference"])


def main() -> None:
    """Entry-point for the ``neural-forge-serve`` CLI command."""
    uvicorn.run("neural_forge.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
