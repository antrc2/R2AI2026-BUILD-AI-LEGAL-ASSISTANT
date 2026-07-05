import os
import json
import re
from typing import List, Dict, Any, Generator, Optional
from services.Chat import ChatService
from services.Search import SearchService

# Định nghĩa Tool cho LLM
SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_referenced_document",
            "description": (
                "Tìm kiếm nội dung cụ thể trong một văn bản pháp luật được trích dẫn. "
                "Sử dụng KHI VÀ CHỈ KHI ngữ cảnh hiện tại nhắc đến một văn bản khác (vd: Luật X, Thông tư Y) "
                "và bạn BẮT BUỘC cần chi tiết từ văn bản đó để trả lời chính xác câu hỏi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ref": {
                        "type": "string",
                        "description": "Số hiệu văn bản pháp luật ĐẦY ĐỦ (ví dụ: '36/2015/QĐ-TTg'). KHÔNG điền [1], [2].",
                    },
                    "dieu_filter": {
                        "type": "string",
                        "description": "(Tùy chọn) Chỉ ghi số điều, ví dụ 'Điều 74'.",
                    },
                    "khoan_filter": {
                        "type": "string",
                        "description": "(Tùy chọn) Chỉ ghi số khoản, ví dụ 'Khoản 3'.",
                    },
                    "content_query": {
                        "type": "string",
                        "description": "(Bắt buộc) Từ khóa hoặc chủ đề cần tìm trong văn bản đó.",
                    },
                },
                "required": ["doc_ref", "content_query"],
            },
        },
    }
]

SUB_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sub_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["queries"],
            "additionalProperties": False
        }
    }
}


class RAGPipeline:
    def __init__(self):
        self.chat_service = ChatService()
        self.search_service = SearchService()
        # Giới hạn số chunk tối đa trong context để tránh tràn token
        self.MAX_CONTEXT_CHUNKS = 50
        self.MAX_TOOL_ITERATIONS = 3

    def _parse_sub_queries(self, content: str) -> List[str]:
        """Parse JSON response từ LLM để lấy danh sách sub-queries."""
        try:
            clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()
            data = json.loads(clean_content)
            queries = data.get("queries", [])
            if isinstance(queries, list) and len(queries) > 0:
                return queries
            return [content]
        except (json.JSONDecodeError, Exception):
            return [content]

    def _format_context(self, docs: List[Dict]) -> str:
        """Format context thành dạng [1]: content, [2]: content..."""
        if not docs:
            return "Không có thông tin ngữ cảnh nào."
        return "\n\n".join([f"[{i + 1}]: {d.get('content', '')}" for i, d in enumerate(docs)])

    def _deduplicate_docs(self, docs: List[Dict]) -> List[Dict]:
        """Loại bỏ các document trùng lặp dựa trên chunk_id."""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = doc.get('chunk_id') or hash(doc.get('content', ''))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def _flatten_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Nối toàn bộ hội thoại (role + content) thành 1 khối text,
        dùng để phân tích sub-query và semantic search."""
        role_labels = {"user": "Người dùng", "assistant": "Trợ lý", "system": "Hệ thống"}
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            if not content:
                continue
            label = role_labels.get(role, role)
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def _get_last_user_question(self, messages: List[Dict[str, str]]) -> str:
        """Lấy câu hỏi mới nhất của user, dùng để log / fallback."""
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def _build_context_message(self, context_docs: List[Dict]) -> Dict[str, str]:
        """Tạo 1 system/user message chứa ngữ cảnh + quy tắc trích dẫn,
        được chèn vào NGAY TRƯỚC lượt hội thoại của user để LLM luôn thấy
        context mới nhất mà không phá vỡ cấu trúc nhiều lượt hội thoại."""
        context_text = self._format_context(context_docs)
        return {
            "role": "system",
            "content": f"""Dựa trên ngữ cảnh pháp lý sau để trả lời câu hỏi mới nhất của người dùng trong hội thoại:
{context_text}

QUY TẮC TRÍCH DẪN BẮT BUỘC:
- Mọi thông tin lấy từ ngữ cảnh đều phải trích dẫn nguồn.
- Sử dụng CHÍNH XÁC định dạng [N] (ví dụ: [1], [2], [3]).
- KHÔNG thêm khoảng trắng (không dùng [ 1 ]), KHÔNG dùng định dạng khác.
- Đặt mã trích dẫn ở cuối câu hoặc cuối ý tương ứng.
- Nếu ngữ cảnh NHẮC ĐẾN một văn bản khác (vd: "theo Luật X") và bạn CẦN chi tiết từ văn bản đó để trả lời
  chính xác, hãy gọi tool `search_referenced_document` thay vì trả lời ngay.
- Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin.
- Hãy tham khảo các lượt hội thoại trước đó (nếu có) để hiểu đúng ý người dùng, nhưng chỉ trích dẫn [N]
  cho thông tin lấy từ ngữ cảnh pháp lý ở trên.""",
        }

    def process(self, messages: List[Dict[str, str]], stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Pipeline xử lý chính.

        messages: lịch sử hội thoại dạng [{"role": "user"/"assistant", "content": "..."}]
        theo đúng thứ tự thời gian, không cần chứa system prompt (pipeline tự thêm).
        """

        system_prompt = (
            "Bạn là trợ lý pháp lý thông minh. Hãy trả lời chính xác, chuyên nghiệp dựa trên ngữ cảnh được cung cấp. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin."
        )

        # Lọc bỏ mọi system message người dùng gửi lên (pipeline tự quản lý system prompt)
        conversation = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        if not conversation:
            yield {"step": "answer", "status": "error", "data": {"error": "Không có nội dung hội thoại hợp lệ."}}
            return

        question = self._get_last_user_question(conversation)
        conversation_text = self._flatten_conversation(conversation)

        # ==========================================
        # BƯỚC 1: SUB-QUERY (Phân tích câu hỏi, dựa trên TOÀN BỘ hội thoại)
        # ==========================================
        yield {"step": "sub_queries", "status": "processing", "data": None}

        sub_query_prompt = (
            f"Hãy phân tích đoạn hội thoại sau, tập trung vào ý định mới nhất của người dùng, "
            f"và tách thành các ý nhỏ (sub-queries) độc lập để tìm kiếm thông tin hiệu quả hơn. "
            f"Trả về kết quả dưới dạng JSON thuần túy với key 'queries'.\n\n"
            f"--- Hội thoại ---\n{conversation_text} /no_think"
        )

        try:
            sub_query_response = ""
            # stream=False -> ChatService yield đúng 1 lần: response.choices[0].message
            for message in self.chat_service.generate_response(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sub_query_prompt},
                ],
                response_format=SUB_QUERY_SCHEMA,
                stream=False,
            ):
                if isinstance(message, dict) and "error" in message:
                    raise Exception(message["error"])
                sub_query_response = getattr(message, "content", "") or ""

            sub_queries = self._parse_sub_queries(sub_query_response)
        except Exception as e:
            print(f"Lỗi khi parse sub-queries: {e}")
            sub_queries = [question]

        yield {"step": "sub_queries", "status": "done", "data": {"queries": sub_queries}}

        # ==========================================
        # BƯỚC 2: SEARCH (Semantic Search ban đầu)
        # ==========================================
        yield {"step": "retrieval", "status": "processing", "data": None}

        retrieved_docs = []
        for sq in sub_queries:
            try:
                docs = self.search_service.semantic_search(query=sq, top_k=15)
                retrieved_docs.extend(docs)
            except Exception as e:
                print(f"Lỗi search cho query '{sq}': {e}")

        unique_docs = self._deduplicate_docs(retrieved_docs)
        context_docs = unique_docs[:self.MAX_CONTEXT_CHUNKS]

        yield {
            "step": "retrieval",
            "status": "done",
            "data": {"count": len(context_docs)},
        }

        # ==========================================================
        # BƯỚC 3+4 (GỘP): LLM STREAM — vừa quyết định tool call vừa
        # trả lời trực tiếp trong CÙNG một lần gọi, giống code mẫu.
        # Lặp tối đa MAX_TOOL_ITERATIONS lần nếu LLM liên tục gọi tool.
        # ==========================================================
        # Cấu trúc: [system prompt, system context+quy tắc trích dẫn, ...toàn bộ hội thoại gốc]
        # Giữ nguyên multi-turn để LLM hiểu đúng mạch hội thoại, thay vì gộp hết vào 1 user message.
        llm_messages = [
            {"role": "system", "content": system_prompt},
            self._build_context_message(context_docs),
            *conversation,
        ]

        full_answer = ""
        citation_map: Dict[str, Any] = {str(i + 1): d for i, d in enumerate(context_docs)}

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            did_tool_call = False
            did_content = False

            # Buffer để gom các mảnh tool_call arguments bị chia nhỏ qua nhiều chunk
            # key = index của tool call trong response (OpenAI có thể trả nhiều tool_calls song song)
            tool_call_buffers: Dict[int, Dict[str, Any]] = {}

            try:
                response_stream = self.chat_service.generate_response(
                    llm_messages,
                    tools=SEARCH_TOOLS,
                    stream=True,
                )
            except Exception as e:
                yield {"step": "answer", "status": "error", "data": {"error": str(e)}}
                return

            if iteration == 0:
                yield {"step": "tool_call", "status": "processing", "data": None}
                yield {"step": "answer", "status": "start", "data": None}

            try:
                for chunk in response_stream:
                    # ChatService yield {"error": ...} thay vì raise khi có lỗi ở giữa stream
                    if isinstance(chunk, dict) and "error" in chunk:
                        raise Exception(chunk["error"])
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta

                    # --- Trả lời trực tiếp (không cần tool) ---
                    if getattr(delta, "content", None):
                        did_content = True
                        piece = delta.content
                        full_answer += piece
                        yield {
                            "step": "answer",
                            "status": "streaming",
                            "data": {
                                "chunk": piece,
                                "citations": citation_map,
                            },
                        }

                    # --- Tool call (có thể tới theo từng mảnh nhỏ) ---
                    if getattr(delta, "tool_calls", None):
                        did_tool_call = True
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_buffers:
                                tool_call_buffers[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            buf = tool_call_buffers[idx]
                            if tc_delta.id:
                                buf["id"] = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments

            except Exception as e:
                yield {"step": "answer", "status": "error", "data": {"error": str(e)}}
                return

            # Nếu vòng này LLM không gọi tool -> đã trả lời xong, thoát loop
            if not did_tool_call:
                break

            # ---- Xử lý các tool call đã gom được ----
            assistant_tool_calls = []
            for idx in sorted(tool_call_buffers.keys()):
                buf = tool_call_buffers[idx]
                assistant_tool_calls.append({
                    "id": buf["id"],
                    "type": "function",
                    "function": {
                        "name": buf["name"],
                        "arguments": buf["arguments"],
                    },
                })

            # Thêm assistant message chứa tool_calls vào history (bắt buộc theo chuẩn OpenAI)
            llm_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })

            for tc in assistant_tool_calls:
                if tc["function"]["name"] != "search_referenced_document":
                    # tool lạ, bỏ qua an toàn
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Tool không được hỗ trợ.",
                    })
                    continue

                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                yield {"step": "tool_call", "status": "detected", "data": {"args": args}}

                try:
                    extra_docs = self.search_service.doc_ref_search(
                        query=args.get("content_query", question),
                        doc_ref=args.get("doc_ref"),
                        article_filter=args.get("dieu_filter"),
                        clause_filter=args.get("khoan_filter"),
                        top_k=15,
                    )
                except Exception as e:
                    print(f"Lỗi thực thi tool: {e}")
                    extra_docs = []
                    yield {"step": "tool_call", "status": "error", "data": {"error": str(e)}}

                if extra_docs:
                    context_docs = self._deduplicate_docs(context_docs + extra_docs)[:self.MAX_CONTEXT_CHUNKS]
                    citation_map = {str(i + 1): d for i, d in enumerate(context_docs)}
                    yield {"step": "tool_call", "status": "executed", "data": {"found_count": len(extra_docs)}}
                    tool_result_content = (
                        f"Đã tìm thấy {len(extra_docs)} đoạn trích từ văn bản {args.get('doc_ref')}. "
                        f"Ngữ cảnh đầy đủ đã được cập nhật ở lượt tiếp theo."
                    )
                else:
                    yield {"step": "tool_call", "status": "executed", "data": {"found_count": 0, "message": "Không tìm thấy thông tin"}}
                    tool_result_content = f"Không tìm thấy thông tin bổ sung trong văn bản {args.get('doc_ref')}."

                llm_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result_content,
                })

            # Cập nhật lại phần "ngữ cảnh" cho lượt gọi tiếp theo bằng cách
            # thêm 1 user message mới chứa context đã bổ sung, để model
            # thực sự "nhìn thấy" nội dung mới lấy được (không chỉ là message
            # thông báo suông ở trên).
            llm_messages.append({
                "role": "user",
                "content": (
                    f"Đây là ngữ cảnh đầy đủ đã được cập nhật sau khi tra cứu thêm:\n\n"
                    f"{self._format_context(context_docs)}\n\n"
                    f"Hãy trả lời câu hỏi gốc: {question}\n"
                    f"Nhớ tuân thủ quy tắc trích dẫn [N] như đã nêu. Nếu vẫn còn thiếu thông tin quan trọng "
                    f"và cần tra cứu thêm văn bản khác, hãy tiếp tục gọi tool."
                ),
            })

            yield {"step": "tool_call", "status": "done", "data": None}
            # loop tiếp -> gọi lại LLM với context mới

        yield {
            "step": "answer",
            "status": "done",
            "data": {
                "text": full_answer,
                "citations": citation_map,
                "sources": context_docs,
            },
        }