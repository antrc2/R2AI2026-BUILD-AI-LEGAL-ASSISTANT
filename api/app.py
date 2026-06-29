"""
api/app.py - FastAPI application cho RAG Chatbot
"""

import os
import json
from typing import Optional, Generator, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from services.RAGPipeline import get_pipeline, RAGPipeline

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="API cho chatbot hỏi đáp pháp luật Việt Nam với RAG pipeline",
    version="1.0.0"
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Pipeline singleton
pipeline: Optional[RAGPipeline] = None


def get_or_init_pipeline() -> RAGPipeline:
    """Get or initialize the pipeline."""
    global pipeline
    if pipeline is None:
        data_dir = os.environ.get("DATA_DIR", "data")
        pipeline = get_pipeline(data_dir=data_dir)
    return pipeline


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Trang chủ với giao diện chatbot."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(
    request: Request,
    question: Optional[str] = None,
    stream: Optional[bool] = True
):
    """
    Endpoint chat chính.
    
    Args:
        question: Câu hỏi của người dùng
        stream: Nếu True, trả về SSE stream với từng bước pipeline.
                Nếu False, trả về JSON một lần với kết quả cuối cùng.
    
    Returns:
        StreamingResponse (SSE) hoặc JSONResponse tùy theo tham số stream.
    """
    # Parse request body nếu là JSON
    if question is None:
        try:
            body = await request.json()
            question = body.get("question")
            if stream is True and "stream" in body:
                stream = body.get("stream", True)
        except Exception:
            pass
    
    if not question:
        raise HTTPException(status_code=400, detail="Thiếu câu hỏi (question)")
    
    try:
        rag_pipeline = get_or_init_pipeline()
        
        if stream:
            # Streaming response với SSE
            async def event_generator() -> Generator[dict, None, None]:
                """Generator cho SSE events."""
                for event in rag_pipeline.process(question, stream=True):
                    yield {
                        "event": "pipeline_step",
                        "data": json.dumps(event, ensure_ascii=False)
                    }
                yield {
                    "event": "done",
                    "data": json.dumps({"status": "completed"}, ensure_ascii=False)
                }
            
            return EventSourceResponse(
                event_generator(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming: trả về JSON một lần
            result = None
            for event in rag_pipeline.process(question, stream=False):
                result = event
            
            return JSONResponse(content=result)
    
    except Exception as e:
        if stream:
            # Trong chế độ stream, trả về lỗi qua SSE
            async def error_generator():
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}, ensure_ascii=False)
                }
            return EventSourceResponse(error_generator())
        else:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG Chatbot API is running"}


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
