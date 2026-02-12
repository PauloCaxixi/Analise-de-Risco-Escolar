from __future__ import annotations

import os

from fastapi import FastAPI
from app.routes import router as api_router
from app.web import router as web_router, init_db
from app.models import init_schema

app = FastAPI(
    title="Plataforma Passos Mágicos",
    version="v1.0.0",
    description="Plataforma única: UI + API + ML + gestão.",
)

app.state.secret_key = os.getenv("PLATFORM_SECRET_KEY", "change-me-now")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    init_schema()


app.include_router(web_router)
app.include_router(api_router, prefix="/api")
