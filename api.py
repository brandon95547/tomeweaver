from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .main import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="tomeweaver-api", version="1.0.0")


class RunPipelineRequest(BaseModel):
    text: str | None = None
    input_path: str | None = None
    toc_full_path: str | None = None
    max_chunk_chars: int | None = None


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/run")
async def run_pipeline_endpoint(payload: RunPipelineRequest) -> dict:
    if not payload.text and not payload.input_path:
        raise HTTPException(status_code=400, detail="Provide either `text` or `input_path`.")

    temp_input_path: Path | None = None
    run_input_path = payload.input_path

    try:
        if payload.text is not None:
            temp_input_path = TMP_DIR / f"api_input_{uuid.uuid4().hex}.txt"
            temp_input_path.write_text(payload.text, encoding="utf-8")
            run_input_path = str(temp_input_path)

        result = run_pipeline(
            input_path=run_input_path,
            toc_full_path=payload.toc_full_path,
            max_chunk_chars=payload.max_chunk_chars,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e
    finally:
        if temp_input_path is not None:
            try:
                temp_input_path.unlink(missing_ok=True)
            except Exception:
                pass
