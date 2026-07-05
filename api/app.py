from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Literal
import json
import os
import queue
import threading
import asyncio
from services.RAGPipeline import RAGPipeline

app = FastAPI(title="Legal RAG API")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

pipeline = RAGPipeline()

# Sentinel để báo hiệu generator (chạy trong thread riêng) đã kết thúc
_SENTINEL = object()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


def _run_pipeline_in_thread(conversation: List[dict], stream: bool, out_queue: "queue.Queue"):
    """Chạy pipeline.process() (sync, blocking) trong 1 thread riêng.

    Vì pipeline.process() là generator đồng bộ chứa các lời gọi HTTP blocking
    (LLM sub-query, semantic search, LLM streaming), nếu chạy trực tiếp bằng
    `for event in pipeline.process(...)` bên trong 1 `async def`, mỗi bước
    blocking đó sẽ giữ chặt event loop của Uvicorn, khiến các event đã yield
    trước đó không được flush ra socket ngay -> client thấy "giật cục", chỉ
    nhận được dữ liệu dồn cục khi có 1 đoạn code async khác nhường CPU.

    Chạy toàn bộ generator trong thread riêng và đẩy từng event qua queue.Queue
    (thread-safe) giúp mỗi event được gửi ra ngay khi có, độc lập với việc
    thread đó có đang block ở HTTP call hay không.
    """
    try:
        for event in pipeline.process(messages=conversation, stream=stream):
            out_queue.put(event)
    except Exception as e:
        out_queue.put({"step": "answer", "status": "error", "data": {"error": str(e)}})
    finally:
        out_queue.put(_SENTINEL)


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "messages không được để trống."})

    conversation = [m.model_dump() for m in req.messages]

    if req.stream:
        async def event_generator():
            out_queue: "queue.Queue" = queue.Queue()
            thread = threading.Thread(
                target=_run_pipeline_in_thread,
                args=(conversation, True, out_queue),
                daemon=True,
            )
            thread.start()

            loop = asyncio.get_event_loop()

            try:
                while True:
                    # out_queue.get là lời gọi blocking (sync) -> chạy trong
                    # threadpool executor riêng để KHÔNG chiếm event loop
                    # chính, cho phép Uvicorn flush dữ liệu ra ngay khi có,
                    # thay vì phải đợi cả pipeline chạy xong mới trả 1 lượt.
                    event = await loop.run_in_executor(None, out_queue.get)

                    if event is _SENTINEL:
                        break

                    print(f"Event: {event}")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'step': 'answer', 'status': 'error', 'data': {'error': str(e)}}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                # Tắt buffering ở phía reverse proxy (vd nginx) để SSE không
                # bị giữ lại theo lô trước khi tới trình duyệt.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-stream: chạy trong threadpool để không block event loop chính,
        # cho phép server vẫn phục vụ được request khác song song.
        full_response = {
            "steps": [],
            "final_answer": "",
            "citations": {},
            "sources": []
        }

        def _run_non_stream():
            return list(pipeline.process(messages=conversation, stream=False))

        try:
            loop = asyncio.get_event_loop()
            steps = await loop.run_in_executor(None, _run_non_stream)

            full_response["steps"] = steps
            for event in steps:
                if event["step"] == "answer" and event["status"] == "done":
                    full_response["final_answer"] = event["data"].get("text", "")
                    full_response["citations"] = event["data"].get("citations", {})
                    full_response["sources"] = event["data"].get("sources", [])

            return JSONResponse(content=full_response)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    return {"status": "ok"}