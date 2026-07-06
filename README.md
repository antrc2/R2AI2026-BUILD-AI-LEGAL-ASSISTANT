

# RAG Chatbot API - Hỏi đáp Pháp luật Việt Nam

## Demo
https://github.com/user-attachments/assets/dccc32c0-e9b4-414a-aecb-b64a7bd6383b


## Giới thiệu

Dự án cung cấp API chatbot hỏi đáp về pháp luật Việt Nam sử dụng kiến trúc RAG (Retrieval-Augmented Generation), hỗ trợ hội thoại nhiều lượt (multi-turn) và streaming theo thời gian thực từng bước xử lý của pipeline (phân tích câu hỏi → tìm kiếm tài liệu → tra cứu chéo văn bản → sinh câu trả lời có trích dẫn).

Dự án được phát triển trong khuôn khổ cuộc thi **[R2AI 2026 — Build AI Legal Assistant](https://leaderboard.aiguru.com.vn/competitions/13/)**, tổ chức bởi BM25 Baseline / AI Guru.

## Kết quả trên bảng xếp hạng (Kiểm thử riêng)

Bảng dưới trích từ leaderboard chính thức của BTC (hạng mục **Kiểm thử riêng**, top 10 tại thời điểm chốt), dùng để đối chiếu vị trí và các chỉ số của đội **Bee IT** so với các đội dẫn đầu khác.

| # | Người tham gia | Ngày | ID | FINAL SCORE | Articles F2-Macro | Docs F2-Macro | Avg QA | Articles Precision | Articles Recall | Docs Precision | Docs Recall | Chính xác nội dung | Đầy đủ & toàn diện | Thực tiễn & áp dụng | Rõ ràng & dễ hiểu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Hung&Fong | 2026-07-02 22:21 | 2210 | **0.6437** | **0.6552** | **0.763** | 0.513 | 0.5654 | 0.722 | 0.6856 | 0.8184 | 0.3836 | 0.3739 | 0.3581 | 0.9366 |
| 2 | Trần Anh Tú | 2026-07-02 17:51 | 2198 | 0.6408 | 0.6117 | 0.7076 | 0.6029 | 0.4548 | 0.7202 | 0.5371 | 0.8217 | 0.4883 | 0.4368 | 0.5166 | 0.9701 |
| 3 | thanhkhauson | 2026-07-02 21:30 | 2204 | 0.6299 | 0.5774 | 0.6062 | **0.7062** | 0.4724 | 0.6629 | 0.4598 | 0.7152 | **0.5886** | **0.543** | **0.7248** | 0.9685 |
| 4 | mscai | 2026-07-02 22:39 | 2212 | 0.6291 | 0.5733 | 0.734 | 0.5799 | 0.5561 | 0.6317 | **0.6944** | 0.7825 | 0.5082 | 0.3812 | 0.4573 | **0.9727** |
| **5** | **Bee IT** | 2026-07-01 13:14 | 2155 | 0.6053 | 0.5612 | 0.6737 | 0.581 | 0.3443 | **0.7694** | 0.4795 | **0.8465** | 0.5517 | 0.4681 | 0.3397 | 0.9643 |
| 6 | Nguyễn Văn Nghiêm | 2026-07-01 09:49 | 2144 | 0.5983 | 0.6008 | 0.6759 | 0.5182 | 0.5473 | 0.6639 | 0.5782 | 0.7573 | 0.437 | 0.4044 | 0.2586 | **0.9727** |
| 7 | Next Gen 2026 | 2026-07-01 22:46 | 2162 | 0.554 | 0.54 | 0.6656 | 0.4565 | 0.4989 | 0.5826 | 0.5639 | 0.7287 | 0.4124 | 0.2868 | 0.2444 | 0.8823 |
| 8 | Agentic Builders Lê Trí Luận | 2026-07-02 19:37 | 2200 | 0.5391 | 0.5065 | 0.6679 | 0.4428 | **0.5721** | 0.5268 | 0.6887 | 0.6841 | 0.4001 | 0.2693 | 0.227 | 0.875 |
| 9 | FAI Team | 2026-07-02 17:24 | 2195 | 0.5211 | 0.5547 | 0.5333 | 0.4752 | 0.5279 | 0.6069 | 0.4729 | 0.5837 | 0.3921 | 0.2981 | 0.3004 | 0.9103 |
| 10 | tqd | 2026-07-01 11:55 | 2150 | 0.5182 | 0.51 | 0.5921 | 0.4524 | 0.5617 | 0.521 | 0.6358 | 0.5981 | 0.4203 | 0.3109 | 0.1998 | 0.8789 |
 
> Đội **Bee IT** (dự án này) xếp **hạng 5/54** ở thời điểm chốt bảng trên, với `FINAL SCORE = 0.6053`. Điểm mạnh nằm ở **Docs Recall (0.8465)** và **Articles Recall (0.7694)**

## Cấu trúc dự án

```
project/
├── app.py                     # FastAPI application (endpoint /chat, /health, /)
├── templates/
│   └── index.html             # Giao diện chatbot web (vanilla JS + SSE + sidebar tài liệu)
├── services/
│   ├── __init__.py
│   ├── Chat.py                # Client Chat (OpenAI-compatible), embedding, rerank
│   ├── OpenAIExtended.py       # OpenAI client mở rộng (thêm endpoint reranker)
│   ├── Search.py               # Semantic search (FAISS) + tra cứu theo doc_ref
│   └── RAGPipeline.py          # Pipeline RAG chính: sub-query → retrieval → context_ready → tool-call → answer
├── data/                       # FAISS index và các map dữ liệu (tải riêng)
│   ├── faiss.index
│   ├── faiss_id_map.json
│   ├── chunk_map.json
│   ├── article_index_map.json
│   └── chunks.json
├── requirements.txt
├── main.py                     # Entry point chạy server (uvicorn)
└── README.md                   # Tài liệu này
```

> Điều chỉnh đường dẫn `app.py`/`templates/` ở trên cho khớp với cách bạn tổ chức thư mục thực tế (ví dụ đặt trong `api/`); `Jinja2Templates` trong `app.py` trỏ tới thư mục `templates` **cùng cấp** với file `app.py`.

## Kiến trúc pipeline

Khác với cách xử lý 2 lượt gọi LLM tách rời (1 lần quyết định tool call, 1 lần sinh câu trả lời), `RAGPipeline` gộp bước quyết định tool-call và sinh câu trả lời vào **một luồng streaming duy nhất**: LLM vừa có thể trả lời trực tiếp (`delta.content`) vừa có thể gọi tool (`delta.tool_calls`) ngay trong cùng 1 lần gọi API, giảm độ trễ và giữ nguyên ngữ cảnh hội thoại giữa các bước.

```
Hội thoại (messages: [user, assistant, user, ...])
  │
  ▼
[Bước 1] Sub-query
  └─► Gộp toàn bộ hội thoại thành 1 khối text
  └─► LLM (non-stream, structured output, /no_think) tách câu hỏi thành các sub-query
      * Mỗi sub-query PHẢI tự đầy đủ ngữ cảnh (giữ nguyên chủ thể/điều kiện của câu hỏi gốc)
      * Nếu các ý không thể tách rời mà không mất nghĩa → trả về đúng 1 sub-query
  │
  ▼
[Bước 2] Retrieval
  └─► Semantic search (FAISS) cho từng sub-query
  └─► Deduplicate theo chunk_id, giới hạn MAX_CONTEXT_CHUNKS
  │
  ▼
[Bước 2.5] Context Ready
  └─► Phát ngay citation_map + danh sách context_docs vừa retrieval được
  └─► Cho phép frontend hiển thị sidebar "Tài liệu tham khảo" TRƯỚC khi LLM bắt đầu sinh câu trả lời
  │
  ▼
[Bước 3+4] Answer (streaming, gộp làm 1 lần gọi LLM, /no_think trên mọi user message)
  └─► Đưa [system prompt, context + quy tắc trích dẫn, ...toàn bộ hội thoại gốc] vào LLM (stream=True)
  └─► Nếu LLM sinh nội dung (delta.content) → stream trực tiếp ra client
  └─► Nếu LLM gọi tool `search_referenced_document` (delta.tool_calls)
        → tra cứu thêm chunk theo doc_ref/điều/khoản cụ thể
        → cập nhật lại context, phát lại "context_ready" với citation_map mới
        → gọi lại LLM (tối đa MAX_TOOL_ITERATIONS lần)
  │
  ▼
Trả về: câu trả lời kèm citation [N] + citation_map (map số thứ tự → chunk nguồn)
```

Toàn bộ các lệnh gọi LLM trong bước 3+4 chạy nối tiếp trong **cùng một danh sách `messages`** (đúng chuẩn tool-calling của OpenAI: `assistant` message chứa `tool_calls` → `tool` message chứa kết quả) để LLM giữ được ngữ cảnh xuyên suốt qua các vòng tra cứu.

### Vì sao có bước "Context Ready" tách riêng?

Ban đầu, dữ liệu `citations`/`sources` được gộp phát cùng lúc với `step: "answer", status: "done"` ngay sau retrieval — nhưng đó cũng chính là tín hiệu **kết thúc câu trả lời** ở cuối luồng. Dùng trùng `answer/done` ở giữa pipeline khiến frontend hiểu nhầm là câu trả lời đã xong ngay từ đầu (trong khi `text` chưa hề tồn tại). Vì vậy, sự kiện chuẩn bị ngữ cảnh được tách thành step riêng: `context_ready`, không đụng vào logic xử lý của `answer`. Event này được phát:

- Ngay sau khi retrieval ban đầu hoàn tất (trước khi gọi LLM sinh câu trả lời).
- Mỗi lần tool `search_referenced_document` tìm thêm được tài liệu mới (context được cập nhật).

### `/no_think` — tắt chế độ suy luận (thinking) của model

Các model dòng Qwen3 chạy qua LM Studio/llama.cpp hỗ trợ chế độ "thinking" (sinh ra khối suy luận trước khi trả lời), có thể bật/tắt bằng hậu tố đặc biệt trong prompt. Pipeline tự động thêm `/no_think` vào cuối content của **mọi message có `role="user"`** trước khi gửi lên LLM (áp dụng cho cả bước sub-query lẫn bước sinh câu trả lời chính), thông qua helper `_with_no_think()`:

- Không sửa đổi message gốc trong lịch sử hội thoại (`llm_messages`) — chỉ áp dụng lên bản sao dùng để gọi API, tránh `/no_think` bị lặp lại nhiều lần qua các vòng lặp tool-call.
- Tự kiểm tra để không thêm trùng nếu content đã kết thúc sẵn bằng `/no_think`.
- Không áp dụng cho `system`, `assistant`, `tool` messages.

Mục đích: giảm độ trễ và tránh model sinh ra khối suy luận dài dòng không cần thiết trong một pipeline vốn đã nhiều bước tuần tự (sub-query → retrieval → tool-call → answer).

## Cài đặt

### Tải dữ liệu (thư mục `data/`)

Repo **không** chứa sẵn thư mục `data/` (FAISS index, chunks, metadata). Tải archive đã đóng gói từ Hugging Face và giải nén vào thư mục gốc của project.

**Linux / macOS:**

```bash
curl -L https://huggingface.co/AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT/resolve/main/data.zip -o data.zip
unzip -o data.zip -d .
```

**Windows PowerShell:**

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT/resolve/main/data.zip" -OutFile "data.zip"
Expand-Archive -Path ".\data.zip" -DestinationPath "." -Force
```

Kiểm tra sau khi giải nén:

```bash
ls data
# hoặc Windows:
Get-ChildItem .\data
```

Cần thấy đủ 4 file: `faiss.index`, `faiss_id_map.json`, `chunk_map.json`, `article_index_map.json` (và `chunks.json` nếu muốn dùng `embed_text` gốc thay vì rebuild từ metadata).

**Nguồn dữ liệu:** kho ngữ liệu được crawl và xử lý từ văn bản pháp luật Việt Nam (luật, nghị định, thông tư...), được chunk theo cấu trúc Điều/Khoản/Điểm rồi embed bằng `Qwen3-Embedding-0.6B`. Toàn bộ archive `data.zip` ở trên là dữ liệu đã qua xử lý (post-processing), sẵn sàng nạp trực tiếp vào FAISS.

### Yêu cầu hệ thống

- Python 3.10+
- Các service bên ngoài (API tương thích OpenAI):
  - **Chat LLM** — hỗ trợ streaming + function calling (khuyến nghị dòng Qwen3 để tương thích `/no_think`)
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
pydantic
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
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server chạy tại `http://localhost:8000`.

## API Endpoints

### 1. `GET /` — Trang chủ chatbot

Trả về giao diện web (`templates/index.html`) để tương tác trực tiếp qua trình duyệt, hiển thị pipeline steps theo thời gian thực và sidebar tài liệu tham khảo.

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

Client cần tự lưu và gửi lại toàn bộ `messages` ở mỗi lượt gọi (bao gồm cả các câu trả lời `assistant` trước đó) để giữ ngữ cảnh hội thoại nhiều lượt. Frontend đi kèm (`templates/index.html`) luôn gọi với `stream: true` (không còn cho phép tắt streaming ở giao diện).

**Response khi `stream=true` (Server-Sent Events):**

Mỗi dòng có định dạng `data: {...}\n\n`, kết thúc bằng `data: [DONE]\n\n`. Các event theo đúng thứ tự pipeline:

```json
{"step": "sub_queries", "status": "processing", "data": null}
{"step": "sub_queries", "status": "done", "data": {"queries": ["...", "..."]}}

{"step": "retrieval", "status": "processing", "data": null}
{"step": "retrieval", "status": "done", "data": {"count": 18}}

{"step": "context_ready", "status": "done", "data": {
  "citations": {"1": {"content": "...", "metadata": {}}, "2": {}},
  "sources": [{"chunk_id": "...", "content": "...", "metadata": {}}]
}}

{"step": "tool_call", "status": "processing", "data": null}
{"step": "answer", "status": "start", "data": null}

{"step": "tool_call", "status": "detected", "data": {"args": {"doc_ref": "04/2017/QH14", "content_query": "ưu đãi thuế VAT"}}}
{"step": "tool_call", "status": "executed", "data": {"found_count": 4}}
{"step": "context_ready", "status": "done", "data": {"citations": {"...": {}}, "sources": [{}]}}
{"step": "tool_call", "status": "done", "data": null}

{"step": "answer", "status": "streaming", "data": {"chunk": "Theo quy định tại", "citations": {"1": {"content": "...", "metadata": {}}, "2": {}}}}
{"step": "answer", "status": "streaming", "data": {"chunk": " Điều 5 [1]...", "citations": {}}}

{"step": "answer", "status": "done", "data": {
  "text": "Toàn bộ câu trả lời hoàn chỉnh, có trích dẫn [1], [2]...",
  "citations": {"1": {"content": "...", "metadata": {}}, "2": {}}
}}
```

Nếu xảy ra lỗi ở bất kỳ bước nào: `{"step": "answer", "status": "error", "data": {"error": "..."}}` (hoặc `{"step": "tool_call", "status": "error", "data": {"error": "..."}}` nếu lỗi xảy ra riêng khi thực thi tool tra cứu chéo văn bản).

> `citations` là map `{"số thứ tự (string)": chunk object}` tương ứng đúng với các số `[N]` xuất hiện trong text câu trả lời — dùng để frontend render tooltip/sidebar tham chiếu nguồn. Lưu ý: event `answer/done` cuối cùng chỉ trả về `text` + `citations` (không có `sources`) — nếu cần danh sách `sources` đầy đủ, lấy từ event `context_ready` gần nhất trong luồng.

**Response khi `stream=false`:**

```json
{
  "steps": [
    {"step": "sub_queries", "status": "done", "data": {"queries": ["..."]}},
    {"step": "retrieval", "status": "done", "data": {"count": 18}},
    {"step": "context_ready", "status": "done", "data": {"citations": {}, "sources": []}},
    {"step": "tool_call", "status": "executed", "data": {"found_count": 4}},
    {"step": "answer", "status": "done", "data": {"text": "...", "citations": {}}}
  ],
  "final_answer": "Toàn bộ câu trả lời hoàn chỉnh...",
  "citations": {"1": {}, "2": {}},
  "sources": []
}
```

> Ở nhánh `stream=false`, `app.py` chỉ đọc `sources` từ event `answer/done` (hiện luôn rỗng vì event này không còn mang `sources`) — nếu cần `sources` đầy đủ trong response non-stream, nên sửa `app.py` để lấy `sources` từ event `context_ready` cuối cùng trong danh sách `steps` thay vì từ `answer/done`.

### 3. `GET /health` — Health check

```json
{ "status": "ok" }
```

## Giao diện web

Truy cập `http://localhost:8000` để dùng giao diện chat có sẵn (`templates/index.html`):

- **Khung chat** — hiển thị lịch sử hội thoại (frontend tự lưu `messages` và gửi lại đầy đủ mỗi lượt), luôn dùng streaming.
- **Nhật ký các bước (step log)** — mỗi bước xử lý (`sub_queries`, `retrieval`, `context_ready`, `tool_call`, `answer`) được **append thành 1 dòng riêng**, không ghi đè lên dòng trước đó, để người dùng xem lại được toàn bộ quá trình xử lý kể cả khi có nhiều vòng lặp tool-call. Sau khi câu trả lời hoàn tất, nhật ký được làm mờ (không ẩn hẳn) để vẫn xem lại được.
- **Sidebar "Tài liệu tham khảo"** — cập nhật ngay khi có event `context_ready` (trước khi LLM trả lời xong), hiển thị danh sách văn bản/điều khoản kèm số hiệu `[N]` tương ứng.
- **Citation badge** — số trích dẫn `[N]` trong câu trả lời được render thành nút tròn màu vàng đồng; bấm vào sẽ cuộn tới thẻ tài liệu tương ứng trong sidebar.
- **Nút "Cuộc trò chuyện mới"** — xóa lịch sử hội thoại phía client và reset sidebar.

## Xử lý streaming ở tầng server

`RAGPipeline.process()` là một generator đồng bộ (chứa các lời gọi HTTP blocking tới LLM/search). Để tránh việc các bước xử lý ban đầu (sub-query, retrieval) bị "kẹt" lại và chỉ được đẩy ra client dồn cục cùng lúc với streaming câu trả lời, `app.py` chạy pipeline trong **một thread riêng** (`_run_pipeline_in_thread`), đẩy từng event qua `queue.Queue`, và phía consumer (`async def event_generator`) đọc queue qua `run_in_executor` — nhờ vậy event loop của FastAPI/Uvicorn không bị block, mỗi bước được flush ra SSE ngay khi hoàn thành.

Response SSE cũng được gửi kèm header `Cache-Control: no-cache` và `X-Accel-Buffering: no` để tránh bị buffer nếu sau này triển khai sau reverse proxy (nginx).

## Cấu hình tham số pipeline

Trong `services/RAGPipeline.py`:

```python
self.MAX_CONTEXT_CHUNKS = 50      # Số chunk tối đa đưa vào context mỗi lượt
self.MAX_TOOL_ITERATIONS = 3       # Số vòng tối đa LLM được gọi lại tool search_referenced_document
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

Pipeline gọi `SearchService.doc_ref_search(...)` để lấy thêm chunk liên quan, cập nhật vào context, phát lại `context_ready` với `citation_map` mới, rồi gọi lại LLM với ngữ cảnh mới — lặp lại tối đa `MAX_TOOL_ITERATIONS` lần trước khi buộc phải tổng hợp câu trả lời cuối cùng.

## Lưu ý

- **Ngữ cảnh hội thoại nhiều lượt**: mỗi lượt gọi `/chat`, backend chỉ đính kèm context (tài liệu RAG) tương ứng cho câu hỏi mới nhất — các câu trả lời `assistant` ở lượt trước vẫn nằm trong `messages` nhưng không kèm theo nguồn trích dẫn cũ. Nếu người dùng hỏi tiếp về một trích dẫn `[N]` ở lượt trước, model có thể không còn "nhìn thấy" đúng nguồn đó trừ khi client tự giữ và gửi lại citation map liên quan.
- **LLM API**: cần hỗ trợ streaming (`stream=True`) và function calling (`tools`) theo chuẩn OpenAI Chat Completions. Nếu model không thuộc dòng Qwen3 (không hỗ trợ hậu tố `/no_think`), chuỗi này sẽ được model coi như văn bản thường và không gây lỗi, nhưng cũng không có tác dụng tắt thinking.
- **Embedding/Reranker**: cần endpoint tương thích OpenAI; reranker là tùy chọn.
- **Dữ liệu**: cần chuẩn bị đầy đủ FAISS index và các file map trong `data/` trước khi chạy.

## Tác giả

Được phát triển bởi **AnTrc2** — team **Bee IT** — trong khuôn khổ cuộc thi [R2AI 2026](https://r2ai.aiguru.com.vn/) — hạng mục xây dựng trợ lý AI hỏi đáp pháp luật Việt Nam.
