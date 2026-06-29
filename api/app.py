from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import json
import os
from services.RAGPipeline import RAGPipeline

app = FastAPI(title="Legal RAG API")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    question: str
    stream: bool = True

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.stream:
        async def event_generator():
            try:
                for event in pipeline.process(question=req.question, stream=True):
                    print(f"Event: {event}")
                    # Format SSE: data: {...}\n\n
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{'error': '{str(e)}'}}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Non-stream mode: Gom tất cả sự kiện lại
        full_response = {
            "steps": [],
            "final_answer": "",
            "sources": []
        }
        
        try:
            for event in pipeline.process(question=req.question, stream=False):
                full_response["steps"].append(event)
                if event["step"] == "answer" and event["status"] == "done":
                    full_response["final_answer"] = event["data"].get("text", "")
                    full_response["sources"] = event["data"].get("sources", [])
            
            return JSONResponse(content=full_response)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "ok"}