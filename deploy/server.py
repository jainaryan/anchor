"""
MindMate MVP server — FastAPI + SSE streaming.
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import MODELS, LOG_DIR
from inference import ModelManager
from logger import ChatLogger
from stats import get_stats


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

manager: ModelManager | None = None
logger: ChatLogger | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, logger
    logger = ChatLogger(LOG_DIR)
    manager = ModelManager()   # blocks until all models are loaded
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def stats():
    return get_stats(LOG_DIR)


@app.get("/api/models")
def get_models():
    return [{"id": m["id"], "name": m["name"]} for m in MODELS]


_MAX_MESSAGES = 200        # max turns kept per request
_MAX_MSG_CHARS = 8000      # max chars per individual message


class ChatRequest(BaseModel):
    session_id: str
    model_id: str
    messages: list[dict]   # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.model_id not in manager.model_ids():
        raise HTTPException(status_code=400, detail="Unknown model_id")

    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    # Validate and sanitize messages
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    sanitized = []
    for m in req.messages[-_MAX_MESSAGES:]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        sanitized.append({"role": role, "content": content[:_MAX_MSG_CHARS]})

    if not sanitized:
        raise HTTPException(status_code=400, detail="No valid messages provided")

    req = ChatRequest(session_id=req.session_id, model_id=req.model_id, messages=sanitized)

    model_name = next((m["name"] for m in MODELS if m["id"] == req.model_id), req.model_id)

    async def generate():
        full_response = ""
        try:
            async for token in manager.stream(req.model_id, req.messages):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"
            if full_response:
                logger.log(req.session_id, req.model_id, model_name, req.messages, full_response)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Serve frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
