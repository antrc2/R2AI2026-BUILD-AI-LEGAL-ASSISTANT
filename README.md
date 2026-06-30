# RAG Pipeline – Trợ lý pháp luật Việt Nam

Hệ thống RAG (Retrieval-Augmented Generation) phục vụ trả lời câu hỏi pháp luật Việt Nam, gồm pipeline 3 giai đoạn (sinh sub-query → tìm kiếm FAISS + rerank → sinh câu trả lời kèm trích dẫn → khai báo tài liệu đã dùng).

## 1. Tổng quan kiến trúc

```
Câu hỏi
  │
  ▼
[Giai đoạn 1] LLM sinh sub-queries (tool calling, ép gọi tool submit_sub_queries)
  │
  ▼
[Giai đoạn 2] FAISS search (top 50) → Rerank (Qwen3-Reranker) → lấy top 30, lọc theo ngưỡng
  │
  ▼
[Giai đoạn 3] LLM sinh câu trả lời dựa trên context, có thể gọi tool search_referenced_document
              để tra cứu thêm văn bản được trích dẫn chéo
  │
  ▼
[Giai đoạn 4] LLM khai báo used_refs (structured output / json_schema)
  │
  ▼
[Giai đoạn 5] Build relevant_docs / relevant_articles → ghi kết quả ra JSON (có resume/checkpoint)
```

Toàn bộ 3 lệnh gọi LLM (sub-query, answer, used_refs) chạy trong **cùng một conversation** (`messages` nối tiếp) để giữ ngữ cảnh xuyên suốt.

## 2. Cấu trúc thư mục

```
.
├── main_v2.py                  # Entry point: chạy toàn bộ pipeline trên tập câu hỏi
├── search_v2.py                 # Module search: FAISS, rerank, search theo doc_ref
├── requirements.txt              # Thư viện Python cần cài
├── R2AIStage1DATA.json           # Input: danh sách câu hỏi (id, question)
├── results.json                  # Output: kết quả (tự sinh, có resume)
└── data/                         # Dữ liệu đã xử lý (tải riêng, xem mục 4)
    ├── chunks.json                # List {chunk_id, embed_text, metadata}
    ├── faiss.index                # FAISS IndexIDMap
    ├── faiss_id_map.json          # {faiss_idx(str) -> chunk_id}
    ├── chunk_map.json             # {chunk_id -> metadata}
    └── article_index_map.json     # {"doc_id|article" -> [faiss_idx,...]}
```

## 3. Yêu cầu môi trường

- Python ≥ 3.10 (cú pháp `list[str]`, `str | None` yêu cầu Python 3.10+)
- GPU khuyến nghị để host model chat/embedding/reranker (CPU vẫn chạy được nhưng chậm)
- Docker (nếu host reranker bằng vLLM)

### 3.1 Cài đặt thư viện Python

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
openai
faiss-cpu
requests
```

> Nếu chạy trên máy có GPU và muốn FAISS tận dụng GPU, có thể thay `faiss-cpu` bằng `faiss-gpu` (tùy theo CUDA driver sẵn có), nhưng `faiss-cpu` là đủ dùng cho pipeline này.

## 4. Tải dữ liệu (thư mục `data/`)

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

## 5. Host các model (LLM, Embedding, Reranker)

Pipeline gọi 3 service riêng biệt qua API tương thích OpenAI:

| Vai trò    | Model khuyến nghị                  | Cách host       | Endpoint mặc định trong code      |
|------------|-------------------------------------|------------------|-------------------------------------|
| Chat (LLM) | `Qwen3-14B-Q8_0.gguf`               | `llama-server`   | `http://localhost:11111/v1`         |
| Embedding  | `Qwen3-Embedding-0.6B-Q8_0.gguf`    | `llama-server`   | `http://127.0.0.1:11113/v1`         |
| Reranker   | `Qwen/Qwen3-Reranker-0.6B`          | vLLM (Docker)    | `http://127.0.0.1:11112/v2/rerank`  |

> Trong code mẫu (`main_v2.py`, `search_v2.py`), chat và embedding chạy trên **2 instance `llama-server` riêng biệt**: chat ở cổng `11111`, embedding ở cổng `11113`. Reranker host riêng bằng vLLM ở cổng `11112`.

### 5.1 Tải checkpoint

```bash
# Chat model
curl -L https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q8_0.gguf \
  -o Qwen3-14B-Q8_0.gguf

# Embedding model
curl -L https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf \
  -o Qwen3-Embedding-0.6B-Q8_0.gguf
```

(Reranker không cần tải GGUF riêng nếu host bằng vLLM — vLLM sẽ tự tải checkpoint `Qwen/Qwen3-Reranker-0.6B` từ Hugging Face khi khởi động container, xem mục 5.3.)

### 5.2 Host Chat model + Embedding model bằng `llama-server`

Cài `llama.cpp` (build sẵn `llama-server`), sau đó chạy:

**Chat model** (cổng 11111):

```bash
./llama-server \
  -m /path/to/Qwen3-14B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 11111 \
  --ctx-size 32768 \
  --n-gpu-layers 999
```

Embedding model host riêng trên một instance `llama-server` khác ở cổng `11113`, chạy với cờ `--embedding`:

```bash
./llama-server \
  -m /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 11113 \
  --embedding \
  --ctx-size 4096
```

> Endpoint embedding cần khớp với `EMBEDDING_URL` trong `search_v2.py` (mặc định `http://127.0.0.1:11113/v1`).

Model name truyền vào API (`CHAT_MODEL`, `EMBEDDING_MODEL` trong code) cần khớp với tên model đang được `llama-server` load — với `llama-server`, thường dùng đúng tên file `.gguf` (ví dụ `Qwen3-14B-Q8_0.gguf`, `Qwen3-Embedding-0.6B-Q8_0.gguf`). Kiểm tra lại bằng:

```bash
curl http://localhost:11111/v1/models
```

### 5.3 Host Reranker bằng vLLM (Docker)

```bash
docker run --rm --gpus all \
  --name qwen3-reranker \
  -p 11112:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Reranker-0.6B \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager
```

Sau khi container chạy, endpoint rerank tương ứng `http://localhost:11112/v2/rerank`, đúng với `RERANKER_URL` đã cấu hình sẵn trong `search_v2.py`.

> Lưu ý: nếu chạy reranker bằng GGUF qua `llama.cpp`/LM Studio thay vì vLLM, cần tự viết lại `rerank()` trong `search_v2.py` cho phù hợp với response schema của server đó — code hiện tại được viết riêng cho response schema dạng `{"results": [{"index": ..., "relevance_score": ...}]}` của vLLM rerank endpoint.

### 5.4 Tổng hợp link tải model

| Thành phần | Link |
|---|---|
| Chat (GGUF) | `https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q8_0.gguf` |
| Embedding (GGUF) | `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf` |
| Reranker (vLLM, khuyến nghị) | `Qwen/Qwen3-Reranker-0.6B` (tự tải khi chạy vLLM) |
| Reranker (GGUF, dự phòng cho llama.cpp/LM Studio) | `https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf` |
| Dữ liệu đã xử lý (`data/`) | `https://huggingface.co/AnTrc2/R2AI2026-BUILD-AI-LEGAL-ASSISTANT/resolve/main/data.zip` |

## 6. Cấu hình endpoint trong code

Trong `main_v2.py`:

```python
CHAT_URL   = "http://localhost:11111/v1"
CHAT_MODEL = "Qwen3-14B-Q8_0.gguf"

INPUT_FILE  = "R2AIStage1DATA.json"
OUTPUT_FILE = "results.json"
```

Trong `search_v2.py`:

```python
EMBEDDING_URL   = "http://127.0.0.1:11113/v1"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBEDDING_DIM = 1024

RERANKER_URL    = "http://127.0.0.1:11112/v2/rerank"
RERANKER_MODEL  = "Qwen/Qwen3-Reranker-0.6B"
```

Sửa các giá trị trên cho khớp với host/port/model thực tế đang chạy trên máy bạn trước khi chạy pipeline.

## 7. Chuẩn bị file input câu hỏi

Tạo file `R2AIStage1DATA.json` ở thư mục gốc, định dạng:

```json
[
  { "id": "1", "question": "Doanh nghiệp nhỏ và vừa được ưu đãi thuế thu nhập doanh nghiệp như thế nào?" },
  { "id": "2", "question": "..." }
]
```

## 8. Chạy pipeline

Sau khi đã:
1. Cài xong thư viện (`pip install -r requirements.txt`)
2. Tải xong `data/`
3. Host xong 3 server (chat, embedding, reranker) và sửa endpoint/model name khớp với code
4. Chuẩn bị `R2AIStage1DATA.json`

Chạy:

```bash
python main_v2.py
```

Pipeline sẽ:
- Load FAISS index + các map từ `data/`
- Lần lượt xử lý từng câu hỏi trong `R2AIStage1DATA.json`
- Ghi checkpoint liên tục vào `results.json` sau mỗi câu (an toàn để dừng/chạy lại giữa chừng — script tự **resume**, bỏ qua các `id` đã có kết quả)

Kết quả mỗi câu hỏi trong `results.json` có dạng:

```json
{
  "id": "1",
  "question": "...",
  "answer": "...",
  "relevant_docs": ["54/2014/QH13|Luật Doanh nghiệp"],
  "relevant_articles": ["54/2014/QH13|Luật Doanh nghiệp|Điều 13"]
}
```

## 9. Ghi chú vận hành / debug

- Log console hiển thị chi tiết từng bước (sinh sub-query, search FAISS, rerank, gọi tool tra cứu thêm, khai báo used_refs) để dễ theo dõi và debug.
- `MAX_RETRIES = 3`: số lần thử lại tối đa cho mỗi lệnh gọi LLM khi lỗi API hoặc model không tuân thủ format yêu cầu (không gọi tool khi bị ép, hoặc trả `used_refs` sai schema).
- `max_tool_rounds = 3` (truyền vào `llm_full_pipeline`): số vòng tối đa LLM được phép gọi tool `search_referenced_document` trước khi bị ép tổng hợp câu trả lời cuối cùng.
- Nếu thiếu `data/chunks.json`, pipeline vẫn chạy được nhưng sẽ tự rebuild text hiển thị từ metadata thay vì dùng `embed_text` gốc (chất lượng context có thể giảm nhẹ).

## 10. Sự cố thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách xử lý |
|---|---|---|
| `FileNotFoundError: Không tìm thấy file index` | Chưa giải nén `data.zip` đúng vị trí | Kiểm tra thư mục `data/` nằm cùng cấp với `main_v2.py` |
| Lỗi kết nối tới `localhost:11111` / `11112` / `11113` | Chưa khởi động `llama-server` (chat/embedding) hoặc container vLLM (reranker) | Kiểm tra `curl http://localhost:11111/v1/models`, `curl http://localhost:11113/v1/models` và `docker ps` |
| Reranker trả lỗi / timeout | Sai `RERANKER_URL`, hoặc container vLLM chưa load xong model | Xem log container: `docker logs qwen3-reranker` |
| Model không gọi tool dù bị ép (`tool_choice="required"`) | Model chat không hỗ trợ tốt function calling, hoặc context quá dài | Thử model lớn hơn, hoặc giảm `ctx-size`/độ dài context đầu vào |