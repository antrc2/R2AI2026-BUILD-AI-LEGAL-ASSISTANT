

# RAG Chatbot API - Hỏi đáp Pháp luật Việt Nam

## Demo
https://github.com/user-attachments/assets/dccc32c0-e9b4-414a-aecb-b64a7bd6383b

## Giới thiệu

Dự án cung cấp API chatbot hỏi đáp về pháp luật Việt Nam sử dụng kiến trúc RAG (Retrieval-Augmented Generation), hỗ trợ hội thoại nhiều lượt (multi-turn) và streaming theo thời gian thực từng bước xử lý của pipeline (phân tích câu hỏi → tìm kiếm tài liệu → tra cứu chéo văn bản → sinh câu trả lời có trích dẫn).

## Cấu trúc dự án

```
project/
├── api/
│   ├── __init__.py
│   ├── app.py                # FastAPI application (endpoint /chat, /health, /)
│   └── templates/
│       └── index.html        # Giao diện chatbot web (vanilla JS + SSE)
├── services/
│   ├── __init__.py
│   ├── Chat.py                # Client Chat (OpenAI-compatible), embedding, rerank
│   ├── OpenAIExtended.py      # OpenAI client mở rộng (thêm endpoint reranker)
│   ├── Search.py              # Semantic search (FAISS) + tra cứu theo doc_ref
│   └── RAGPipeline.py         # Pipeline RAG chính: sub-query → retrieval → tool-call → answer
├── data/                      # FAISS index và các map dữ liệu (tải riêng)
│   ├── faiss.index
│   ├── faiss_id_map.json
│   ├── chunk_map.json
│   ├── article_index_map.json
│   └── chunks.json
├── requirements.txt
├── main.py                    # Entry point chạy server (uvicorn)
└── README.md                  # Tài liệu này
```

## Kiến trúc pipeline

Khác với cách xử lý 2 lượt gọi LLM tách rời (1 lần quyết định tool call, 1 lần sinh câu trả lời), `RAGPipeline` gộp bước quyết định tool-call và sinh câu trả lời vào **một luồng streaming duy nhất**: LLM vừa có thể trả lời trực tiếp (`delta.content`) vừa có thể gọi tool (`delta.tool_calls`) ngay trong cùng 1 lần gọi API, giảm độ trễ và giữ nguyên ngữ cảnh hội thoại giữa các bước.

```
Hội thoại (messages: [user, assistant, user, ...])
  │
  ▼
[Bước 1] Sub-query
  └─► Gộp toàn bộ hội thoại thành 1 khối text
  └─► LLM (non-stream, structured output) tách thành các sub-query độc lập
  │
  ▼
[Bước 2] Retrieval
  └─► Semantic search (FAISS) cho từng sub-query
  └─► Deduplicate theo chunk_id, giới hạn MAX_CONTEXT_CHUNKS
  │
  ▼
[Bước 3+4] Answer (streaming, gộp làm 1 lần gọi LLM)
  └─► Đưa [system prompt, context + quy tắc trích dẫn, ...toàn bộ hội thoại gốc] vào LLM (stream=True)
  └─► Nếu LLM sinh nội dung (delta.content) → stream trực tiếp ra client
  └─► Nếu LLM gọi tool `search_referenced_document` (delta.tool_calls)
        → tra cứu thêm chunk theo doc_ref/điều/khoản cụ thể
        → cập nhật lại context, gọi lại LLM (tối đa MAX_TOOL_ITERATIONS lần)
  │
  ▼
Trả về: câu trả lời kèm citation [N] + citation_map (map số thứ tự → chunk nguồn)
```

Toàn bộ các lệnh gọi LLM trong bước 3+4 chạy nối tiếp trong **cùng một danh sách `messages`** (đúng chuẩn tool-calling của OpenAI: `assistant` message chứa `tool_calls` → `tool` message chứa kết quả) để LLM giữ được ngữ cảnh xuyên suốt qua các vòng tra cứu.

## Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- Các service bên ngoài (API tương thích OpenAI):
  - **Chat LLM** — hỗ trợ streaming + function calling
  - **Embedding model**
  - **Reranker** (tùy chọn — nếu không cấu hình, `Chat.py` giữ nguyên thứ tự kết quả semantic search)

### Cài đặt dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` (tối thiểu):

```
fastapi
uvicorn[standard]
jinja2
openai
python-dotenv
faiss-cpu
requests
```

### Chuẩn bị dữ liệu

Đảm bảo thư mục `data/` chứa đủ:
- `faiss.index` — FAISS index đã build
- `faiss_id_map.json` — map FAISS ID → chunk ID
- `chunk_map.json` — metadata của các chunk
- `article_index_map.json` — index map theo Điều/Khoản để phục vụ tool `search_referenced_document`
- `chunks.json` — nội dung text gốc của từng chunk (dùng để build context)

## Cấu hình

`services/Chat.py` đọc cấu hình từ biến môi trường (file `.env` ở thư mục gốc):

```env
# --- Chat model ---
CHAT_MODEL_NAME=qwen3-4b
CHAT_BASE_URL=http://localhost:1234/v1
CHAT_API_KEY=dont need

# --- Embedding model ---
EMBEDDING_MODEL_NAME=text-embedding-qwen3-embedding-0.6b
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_API_KEY=dont need

# --- Reranker (tùy chọn, để trống nếu không dùng) ---
RERANKER_MODEL_NAME=
RERANKER_BASE_URL=
RERANKER_API_KEY=
```

> Nếu để trống 1 trong 3 biến `RERANKER_*`, `ChatService` sẽ không khởi tạo `rerank_client` và tự động giữ nguyên thứ tự tài liệu từ semantic search thay vì rerank.

### Ví dụ host model bằng llama.cpp / vLLM

Nếu chưa có sẵn service LLM/embedding tương thích OpenAI, có thể tự host bằng `llama-server` (llama.cpp):

```bash
# Chat model
./llama-server \
  -m /path/to/model-chat.gguf \
  --host 0.0.0.0 --port 1234 \
  --ctx-size 32768 \
  --n-gpu-layers 999
```

```bash
# Embedding model (cổng riêng, thêm cờ --embedding)
./llama-server \
  -m /path/to/model-embedding.gguf \
  --host 0.0.0.0 --port 1235 \
  --embedding --ctx-size 4096
```

Reranker (tùy chọn) có thể host bằng vLLM, ví dụ với `Qwen3-Reranker-0.6B`:

```bash
docker run --rm --gpus all \
  --name reranker \
  -p 1236:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Reranker-0.6B \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.4 --enforce-eager
```

Sau đó cập nhật `.env` cho khớp cổng/model đang chạy. Kiểm tra nhanh:

```bash
curl http://localhost:1234/v1/models
curl http://localhost:1235/v1/models
```

## Chạy server

```bash
python main.py
```

Hoặc chạy trực tiếp bằng uvicorn:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Server chạy tại `http://localhost:8000`.

## API Endpoints

### 1. `GET /` — Trang chủ chatbot

Trả về giao diện web (`api/templates/index.html`) để tương tác trực tiếp qua trình duyệt, có toggle bật/tắt streaming và hiển thị pipeline steps theo thời gian thực.

### 2. `POST /chat` — Endpoint chat chính

**Request:**

```json
{
  "messages": [
    { "role": "user", "content": "Doanh nghiệp nhỏ và vừa được ưu đãi thuế TNDN như thế nào?" },
    { "role": "assistant", "content": "Theo Luật Hỗ trợ DNNVV..." },
    { "role": "user", "content": "Vậy còn thuế VAT thì sao?" }
  ],
  "stream": true
}
```

**Parameters:**

| Field      | Kiểu             | Bắt buộc | Mô tả                                                                 |
|------------|------------------|----------|------------------------------------------------------------------------|
| `messages` | `List[ChatMessage]` | có     | Toàn bộ lịch sử hội thoại, mỗi phần tử `{"role": "user"\|"assistant", "content": "..."}`. Không cần gửi kèm `system` message — pipeline tự thêm. |
| `stream`   | `bool`           | không (mặc định `true`) | `true`: trả về Server-Sent Events; `false`: trả JSON một lần khi xử lý xong toàn bộ. |

Client cần tự lưu và gửi lại toàn bộ `messages` ở mỗi lượt gọi (bao gồm cả các câu trả lời `assistant` trước đó) để giữ ngữ cảnh hội thoại nhiều lượt.

**Response khi `stream=true` (Server-Sent Events):**

Mỗi dòng có định dạng `data: {...}\n\n`, kết thúc bằng `data: [DONE]\n\n`. Các event theo đúng thứ tự pipeline:

```json
{"step": "sub_queries", "status": "processing", "data": null}
{"step": "sub_queries", "status": "done", "data": {"queries": ["...", "..."]}}

{"step": "retrieval", "status": "processing", "data": null}
{"step": "retrieval", "status": "done", "data": {"count": 18}}

{"step": "tool_call", "status": "processing", "data": null}
{"step": "answer", "status": "start", "data": null}

{"step": "tool_call", "status": "detected", "data": {"args": {"doc_ref": "04/2017/QH14", "content_query": "ưu đãi thuế VAT"}}}
{"step": "tool_call", "status": "executed", "data": {"found_count": 4}}
{"step": "tool_call", "status": "done", "data": null}

{"step": "answer", "status": "streaming", "data": {"chunk": "Theo quy định tại", "citations": {"1": {"content": "...", "metadata": {}}, "2": {}}}}
{"step": "answer", "status": "streaming", "data": {"chunk": " Điều 5 [1]...", "citations": {}}}

{"step": "answer", "status": "done", "data": {
  "text": "Toàn bộ câu trả lời hoàn chỉnh, có trích dẫn [1], [2]...",
  "citations": {"1": {"content": "...", "metadata": {}}, "2": {}},
  "sources": [{"chunk_id": "...", "content": "...", "metadata": {}}]
}}
```

Nếu xảy ra lỗi ở bất kỳ bước nào: `{"step": "answer", "status": "error", "data": {"error": "..."}}`.

> `citations` là map `{"số thứ tự (string)": chunk object}` tương ứng đúng với các số `[N]` xuất hiện trong text câu trả lời — dùng để frontend render tooltip/tham chiếu nguồn.

**Response khi `stream=false`:**

```json
{
  "steps": [
    {"step": "sub_queries", "status": "done", "data": {"queries": ["..."]}},
    {"step": "retrieval", "status": "done", "data": {"count": 18}},
    {"step": "tool_call", "status": "executed", "data": {"found_count": 4}},
    {"step": "answer", "status": "done", "data": {"text": "...", "citations": {}, "sources": []}}
  ],
  "final_answer": "Toàn bộ câu trả lời hoàn chỉnh...",
  "citations": {"1": {}, "2": {}},
  "sources": [{}]
}
```

### 3. `GET /health` — Health check

```json
{ "status": "ok" }
```

## Giao diện web

Truy cập `http://localhost:8000` để dùng giao diện chat có sẵn:

- **Khung chat** — hiển thị lịch sử hội thoại (frontend tự lưu `messages` và gửi lại đầy đủ mỗi lượt)
- **Toggle Streaming** — bật/tắt SSE
- **Steps log** — hiển thị theo thời gian thực từng bước: 🔍 phân tích câu hỏi, 📚 tìm kiếm tài liệu, 🛠️ tra cứu văn bản chéo, ✍️ đang tổng hợp câu trả lời
- **Citation hover** — di chuột vào `[1]`, `[2]`... để xem snippet nguồn
- **Danh sách tài liệu tham khảo** — hiển thị cuối mỗi câu trả lời
- **Nút "Cuộc trò chuyện mới"** — xóa lịch sử hội thoại phía client

## Xử lý streaming ở tầng server

`RAGPipeline.process()` là một generator đồng bộ (chứa các lời gọi HTTP blocking tới LLM/search). Để tránh việc các bước xử lý ban đầu (sub-query, retrieval) bị "kẹt" lại và chỉ được đẩy ra client dồn cục cùng lúc với streaming câu trả lời, `api/app.py` chạy pipeline trong **một thread riêng**, đẩy từng event qua `queue.Queue`, và phía consumer (`async def event_generator`) đọc queue qua `run_in_executor` — nhờ vậy event loop của FastAPI/Uvicorn không bị block, mỗi bước được flush ra SSE ngay khi hoàn thành.

Response SSE cũng được gửi kèm header `Cache-Control: no-cache` và `X-Accel-Buffering: no` để tránh bị buffer nếu sau này triển khai sau reverse proxy (nginx).

## Cấu hình tham số pipeline

Trong `services/RAGPipeline.py`:

```python
self.MAX_CONTEXT_CHUNKS = 20      # Số chunk tối đa đưa vào context mỗi lượt
self.MAX_TOOL_ITERATIONS = 3      # Số vòng tối đa LLM được gọi lại tool search_referenced_document
```

## Tool tra cứu chéo văn bản

Khi LLM phát hiện ngữ cảnh nhắc tới một văn bản khác (ví dụ "theo Luật X", "hướng dẫn tại Thông tư Y") mà cần chi tiết cụ thể để trả lời chính xác, nó có thể tự gọi tool:

```json
{
  "name": "search_referenced_document",
  "arguments": {
    "doc_ref": "36/2015/QĐ-TTg",
    "dieu_filter": "Điều 74",
    "khoan_filter": "Khoản 3",
    "content_query": "điều kiện áp dụng"
  }
}
```

Pipeline gọi `SearchService.doc_ref_search(...)` để lấy thêm chunk liên quan, cập nhật vào context, rồi gọi lại LLM với ngữ cảnh mới — lặp lại tối đa `MAX_TOOL_ITERATIONS` lần trước khi buộc phải tổng hợp câu trả lời cuối cùng.

## Lưu ý

- **Ngữ cảnh hội thoại nhiều lượt**: mỗi lượt gọi `/chat`, backend chỉ đính kèm context (tài liệu RAG) tương ứng cho câu hỏi mới nhất — các câu trả lời `assistant` ở lượt trước vẫn nằm trong `messages` nhưng không kèm theo nguồn trích dẫn cũ. Nếu người dùng hỏi tiếp về một trích dẫn `[N]` ở lượt trước, model có thể không còn "nhìn thấy" đúng nguồn đó trừ khi client tự giữ và gửi lại citation map liên quan.
- **LLM API**: cần hỗ trợ streaming (`stream=True`) và function calling (`tools`) theo chuẩn OpenAI Chat Completions.
- **Embedding/Reranker**: cần endpoint tương thích OpenAI; reranker là tùy chọn.
- **Dữ liệu**: cần chuẩn bị đầy đủ FAISS index và các file map trong `data/` trước khi chạy.

## Tác giả

Được phát triển bởi **AnTrc2** — team **Bee IT** — trong khuôn khổ cuộc thi [R2AI 2026](https://r2ai.aiguru.com.vn/) — hạng mục xây dựng trợ lý AI hỏi đáp pháp luật Việt Nam.
