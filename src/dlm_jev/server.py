"""FastAPI server: POST /v1/decide. Local, single-process; binds to 127.0.0.1 by default."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from dlm_jev.calibration import Calibrator
from dlm_jev.engine import DEFAULT_MODEL, DiffusionReader
from dlm_jev.schema import DecideRequest, DecideResponse

_reader: Optional[DiffusionReader] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup."""
    global _reader
    calib_path = os.environ.get("DLM_JEV_CALIBRATION")
    calibrator = Calibrator.load(Path(calib_path)) if calib_path else None
    _reader = DiffusionReader(
        os.environ.get("DLM_JEV_MODEL", DEFAULT_MODEL),
        calibrator=calibrator,
        extra_reads=int(os.environ.get("DLM_JEV_EXTRA_READS", "0")),
        adapter_path=os.environ.get("DLM_JEV_ADAPTER"),
    )
    yield


app = FastAPI(title="dlm-jev", lifespan=lifespan)


@app.post("/v1/decide", response_model=DecideResponse)
async def decide(request: DecideRequest) -> DecideResponse:
    """Answer typed questions about a state."""
    if _reader is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        return await run_in_threadpool(_reader.decide, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def main() -> None:
    """Entry point: ``uv run dlm_jev-serve``."""
    import uvicorn

    uvicorn.run(app, host=os.environ.get("DLM_JEV_HOST", "127.0.0.1"), port=int(os.environ.get("DLM_JEV_PORT", "8765")))
