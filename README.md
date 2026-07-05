# RAG Chatbot API - Hỏi đáp Pháp luật Việt Nam


## Demo

<video controls width="800">
  <source src="./assets/AI Legal Assistant.mp4.mp4" type="video/mp4">
  Trình duyệt của bạn không hỗ trợ video.
</video>


## Giới thiệu

Dự án cung cấp API chatbot hỏi đáp về pháp luật Việt Nam sử dụng mô hình RAG (Retrieval-Augmented Generation). Hệ thống cho phép người dùng đặt câu hỏi và nhận câu trả lời dựa trên cơ sở dữ liệu văn bản pháp luật, với khả năng streaming từng bước xử lý của pipeline.

## Cấu trúc dự án

```
project/
├── services/
│   ├── __init__.py
│   ├── Chat.py              # (Không cần sửa) Client Chat cơ bản
│   ├── OpenAIExtended.py    # (Không cần sửa) OpenAI client mở rộng
│   ├── Search.py            # (Không cần sửa) Module tìm kiếm cơ bản
│   └── RAGPipeline.py       # ✨ MỚI: Pipeline RAG chính cho API
├── api/
│   ├── __init__.py
│   ├── app.py               # ✨ MỚI: FastAPI application
│   └── templates/
│       └── index.html       # ✨ MỚI: Giao diện chatbot web
├── data/                    # Thư mục chứa FAISS index và maps
│   ├── faiss.index
│   ├── faiss_id_map.json
│   ├── chunk_map.json
│   ├── article_index_map.json
│   └── chunks.json
├── main_only_search.py      # (Giữ nguyên) Code gốc batch processing
├── search_v2.py             # (Giữ nguyên) Module tìm kiếm & rerank
├── requirements.txt         # ✨ Cập nhật thêm dependencies mới
├── main.py                  # Entry point để chạy server
└── README.md                # Tài liệu này
```

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- Các service bên ngoài:
  - LLM API (icllmlib compatible)
  - Embedding API (OpenAI compatible)
  - Reranker API (FastAPI endpoint)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chuẩn bị dữ liệu

Đảm bảo thư mục `data/` chứa các file sau:
- `faiss.index` - FAISS index đã được build
- `faiss_id_map.json` - Map từ FAISS ID sang chunk ID
- `chunk_map.json` - Metadata của các chunks
- `article_index_map.json` - Index map cho articles
- `chunks.json` - Danh sách chunks với embed_text

## Chạy server

```bash
python main.py
```

Server sẽ chạy tại `http://localhost:8000`

## API Endpoints

### 1. GET `/` - Trang chủ chatbot

Trả về giao diện web chatbot để người dùng tương tác trực tiếp.

### 2. POST `/chat` - Endpoint chat chính

**Request:**
```json
{
  "question": "Câu hỏi của bạn",
  "stream": true
}
```

**Parameters:**
- `question` (string, required): Câu hỏi cần trả lời
- `stream` (boolean, optional, default=true): 
  - `true`: Trả về Server-Sent Events (SSE) stream với từng bước pipeline
  - `false`: Trả về JSON một lần với kết quả cuối cùng

**Response (stream=true):**

SSE events với các bước:

1. **sub_queries** - Phân tích câu hỏi thành sub-queries
```json
{
  "step": "sub_queries",
  "status": "processing",
  "message": "Đang phân tích câu hỏi thành sub-queries..."
}
```

```json
{
  "step": "sub_queries",
  "status": "completed",
  "data": {
    "sub_queries": ["query 1", "query 2"],
    "count": 2
  }
}
```

2. **retrieval** - Tìm kiếm tài liệu
```json
{
  "step": "retrieval",
  "status": "completed",
  "data": {
    "total_found": 15,
    "results": [
      {
        "ref_id": 1,
        "doc_num": "04/2017/QH14",
        "doc_name": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
        "article": "Điều 5",
        "score": 0.85,
        "text": "Nội dung chunk..."
      }
    ]
  }
}
```

3. **tool_call** - Kiểm tra sử dụng tool
```json
{
  "step": "tool_call",
  "status": "completed",
  "data": {
    "used_tool": false,
    "message": "Không sử dụng tool ngoài"
  }
}
```

4. **answer** - Sinh câu trả lời
```json
{
  "step": "answer",
  "status": "completed",
  "data": {
    "answer": "Nội dung câu trả lời với [1], [2]...",
    "used_refs": [1, 2],
    "relevant_docs": ["04/2017/QH14|Luật..."],
    "relevant_articles": ["04/2017/QH14|Luật...|Điều 5"],
    "references_info": [
      {
        "ref_id": 1,
        "doc_num": "04/2017/QH14",
        "doc_name": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
        "article": "Điều 5",
        "text": "Nội dung đầy đủ của chunk..."
      }
    ]
  }
}
```

**Response (stream=false):**

```json
{
  "step": "final",
  "status": "completed",
  "data": {
    "answer": "Nội dung câu trả lời với [1], [2]...",
    "used_refs": [1, 2],
    "relevant_docs": ["04/2017/QH14|Luật..."],
    "relevant_articles": ["04/2017/QH14|Luật...|Điều 5"],
    "references_info": [...],
    "pipeline_steps": {
      "sub_queries": ["query 1", "query 2"],
      "retrieval_count": 15
    }
  }
}
```

### 3. GET `/health` - Health check

```json
{
  "status": "ok",
  "message": "RAG Chatbot API is running"
}
```

## Tính năng giao diện web

Khi truy cập `http://localhost:8000`, bạn sẽ thấy:

1. **Khung chat** - Hiển thị lịch sử hội thoại
2. **Toggle Streaming** - Bật/tắt chế độ streaming
3. **Hiển thị pipeline steps** - Khi stream=true, hiển thị từng bước:
   - ⏳ Phân tích câu hỏi (số lượng sub-queries)
   - 🔍 Tìm kiếm tài liệu (số kết quả tìm được)
   - 🛠️ Kiểm tra tool call
   - 💬 Sinh câu trả lời
4. **Citation hover** - Di chuột vào [1], [2] để xem thông tin tài liệu tham khảo
5. **References section** - Danh sách tài liệu tham khảo cuối câu trả lời

## Pipeline xử lý

Quy trình xử lý một câu hỏi:

```
1. Sub-queries
   └─► Phân tích câu hỏi gốc thành các câu hỏi con (nếu cần)
   
2. Retrieval
   └─► FAISS search cho mỗi query
   └─► Gộp candidates vào pool
   └─► Rerank toàn bộ pool theo câu hỏi gốc
   └─► Lọc theo threshold
   
3. Tool Call Check
   └─► Kiểm tra có cần dùng tool ngoài không (hiện tại không dùng)
   
4. Answer Generation
   └─► Build context từ retrieved documents
   └─► LLM sinh câu trả lời với citations [N]
   └─► Trích xuất used_refs từ câu trả lời
   └─► Build relevant_docs và relevant_articles
```

## Configuration

Các tham số có thể điều chỉnh trong `services/RAGPipeline.py`:

```python
PER_QUERY_FAISS_K  = 45    # Số candidate lấy từ FAISS cho mỗi query
GLOBAL_RERANK_CAP  = 200   # Giới hạn pool trước khi rerank
FINAL_CONTEXT_K    = 20    # Số chunk tối đa đưa vào context
RERANK_THRESHOLD   = 0.35  # Ngưỡng lọc sau rerank

ENABLE_FOLLOWUP_LOOKUP = False  # Bật/tắt tra cứu thêm
MAX_FOLLOWUP_ROUNDS    = 2      # Số vòng tra cứu thêm tối đa
FOLLOWUP_TOPK          = 6      # Số chunk lấy thêm mỗi vòng
```

## Lưu ý

- **LLM API**: Cần có service LLM tương thích icllmlib chạy tại `URL_LLM_API`
- **Embedding API**: Cần có service embedding OpenAI-compatible
- **Reranker API**: Cần có reranker FastAPI endpoint
- **Dữ liệu**: Cần chuẩn bị đầy đủ FAISS index và maps trước khi chạy

## Demo

Để demo nhanh:

1. Khởi động server: `python main.py`
2. Mở trình duyệt tại `http://localhost:8000`
3. Nhập câu hỏi và nhấn Gửi
4. Xem kết quả streaming từng bước hoặc JSON tùy chế độ

## Tác giả

Developed for Vietnamese Legal Q&A RAG system." 
