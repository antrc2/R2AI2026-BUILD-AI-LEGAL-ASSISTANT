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
        self.MAX_CONTEXT_CHUNKS = 20 

    def _parse_sub_queries(self, content: str) -> List[str]:
        """Parse JSON response từ LLM để lấy danh sách sub-queries."""
        try:
            # Loại bỏ markdown code block nếu có
            clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()
            data = json.loads(clean_content)
            queries = data.get("queries", [])
            if isinstance(queries, list) and len(queries) > 0:
                return queries
            return [content]
        except (json.JSONDecodeError, Exception):
            # Fallback: coi toàn bộ nội dung là 1 query
            return [content]

    def _format_context(self, docs: List[Dict]) -> str:
        """Format context thành dạng [1]: content, [2]: content..."""
        if not docs:
            return "Không có thông tin ngữ cảnh nào."
        return "\n\n".join([f"[{i+1}]: {d.get('content', '')}" for i, d in enumerate(docs)])

    def _deduplicate_docs(self, docs: List[Dict]) -> List[Dict]:
        """Loại bỏ các document trùng lặp dựa trên chunk_id."""
        seen = set()
        unique_docs = []
        for doc in docs:
            # Ưu tiên dùng chunk_id, nếu không có thì dùng hash của content
            doc_id = doc.get('chunk_id') or hash(doc.get('content', ''))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def process(self, question: str, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Pipeline xử lý chính."""
        
        # Khởi tạo history chat cơ bản
        system_prompt = (
            "Bạn là trợ lý pháp lý thông minh. Hãy trả lời chính xác, chuyên nghiệp dựa trên ngữ cảnh được cung cấp. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin."
        )
        current_messages = [
            {"role": "system", "content": system_prompt},
        ]

        # ==========================================
        # BƯỚC 1: SUB-QUERY (Phân tích câu hỏi)
        # ==========================================
        yield {"step": "sub_queries", "status": "processing", "data": None}
        
        sub_query_prompt = (
            f"Hãy phân tích câu hỏi sau thành các ý nhỏ (sub-queries) độc lập để tìm kiếm thông tin hiệu quả hơn. "
            f"Trả về kết quả dưới dạng JSON thuần túy với key 'queries'.\n\nCâu hỏi: {question} /no_think"
        )
        
        try:
            sub_query_response = ""
            # Gọi non-stream để lấy JSON chắc chắn
            for chunk in self.chat_service.generate_response(
                current_messages + [{"role": "user", "content": sub_query_prompt}], 
                response_format=SUB_QUERY_SCHEMA,
                stream=False
            ):
                # Xử lý cả object và dict tùy vào implementation của ChatService
                if hasattr(chunk, 'content'):
                    sub_query_response = chunk.content
                elif isinstance(chunk, dict):
                    sub_query_response = chunk.get('content', '')
                else:
                    sub_query_response = str(chunk)
            
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
                docs = self.search_service.semantic_search(query=sq, top_k=5)
                retrieved_docs.extend(docs)
            except Exception as e:
                print(f"Lỗi search cho query '{sq}': {e}")

        # Deduplicate và giới hạn số lượng
        unique_docs = self._deduplicate_docs(retrieved_docs)
        initial_context_docs = unique_docs[:self.MAX_CONTEXT_CHUNKS]
        
        yield {
            "step": "retrieval", 
            "status": "done", 
            "data": {"count": len(initial_context_docs)}
        }

        # ==========================================
        # BƯỚC 3: LLM DECISION & TOOL CALLING LOOP
        # ==========================================
        # Thêm ngữ cảnh ban đầu vào prompt để LLM quyết định
        initial_context_text = self._format_context(initial_context_docs)
        
        decision_prompt_content = f"""Dựa trên ngữ cảnh sau:
{initial_context_text}

Câu hỏi gốc: {question}

Yêu cầu xử lý:
1. Nếu ngữ cảnh ĐÃ ĐỦ thông tin để trả lời: Hãy trả lời trực tiếp (không gọi tool).
2. Nếu ngữ cảnh NHẮC ĐẾN một văn bản khác (ví dụ: "theo Luật X", "hướng dẫn tại Thông tư Y") và bạn CẦN chi tiết từ văn bản đó để trả lời chính xác: Hãy gọi tool `search_referenced_document` /no_think.
"""
        
        # Thêm user message vào history
        current_messages.append({"role": "user", "content": decision_prompt_content})
        
        yield {"step": "tool_call", "status": "processing", "data": None}

        final_context_docs = list(initial_context_docs)
        max_tool_iterations = 3 # Tránh loop vô hạn
        
        for iteration in range(max_tool_iterations):
            # Gọi LLM để kiểm tra tool call (Non-stream)
            try:
                response_iter = self.chat_service.generate_response(
                    current_messages,
                    tools=SEARCH_TOOLS,
                    stream=False
                )
                message_result = next(response_iter)
                
                # Kiểm tra xem có tool_calls không
                tool_calls = getattr(message_result, 'tool_calls', None)
                
                if not tool_calls:
                    # Không có tool call, thoát vòng lặp để trả lời
                    break
                
                # Xử lý từng tool call
                for tc in tool_calls:
                    if tc.function.name == "search_referenced_document":
                        try:
                            args = json.loads(tc.function.arguments)
                            yield {"step": "tool_call", "status": "detected", "data": {"args": args}}
                            
                            # Thực hiện tìm kiếm bổ sung
                            extra_docs = self.search_service.doc_ref_search(
                                query=args.get("content_query", question),
                                doc_ref=args.get("doc_ref"),
                                article_filter=args.get("dieu_filter"),
                                clause_filter=args.get("khoan_filter"),
                                top_k=5
                            )
                            
                            if extra_docs:
                                final_context_docs.extend(extra_docs)
                                # Deduplicate lại toàn bộ sau khi thêm mới
                                final_context_docs = self._deduplicate_docs(final_context_docs)[:self.MAX_CONTEXT_CHUNKS]
                                
                                yield {"step": "tool_call", "status": "executed", "data": {"found_count": len(extra_docs)}}
                                
                                # Tạo thông báo kết quả tool để đưa lại vào history cho LLM biết
                                tool_result_content = f"Đã tìm thấy {len(extra_docs)} đoạn trích từ văn bản {args.get('doc_ref')}."
                                # Lưu ý: Một số model yêu cầu format đặc biệt cho tool result, 
                                # ở đây ta giả lập bằng cách thêm vào context hoặc message mới nếu cần.
                                # Cách an toàn nhất là cập nhật lại context trong prompt cuối cùng.
                                
                            else:
                                yield {"step": "tool_call", "status": "executed", "data": {"found_count": 0, "message": "Không tìm thấy thông tin"}}

                        except Exception as e:
                            print(f"Lỗi thực thi tool: {e}")
                            yield {"step": "tool_call", "status": "error", "data": {"error": str(e)}}

                # Nếu muốn hỗ trợ multi-turn tool calling thực sự, ta cần thêm assistant message và tool result vào history.
                # Tuy nhiên, với kiến trúc RAG đơn giản, việc cập nhật lại `final_context_docs` và tạo prompt mới ở bước 4 là đủ.
                # Ta break luôn sau khi lấy tool để sang bước generate answer với context mới.
                break 

            except Exception as e:
                print(f"Lỗi khi gọi LLM decision: {e}")
                break

        yield {"step": "tool_call", "status": "done", "data": None}

        # ==========================================
        # BƯỚC 4: GENERATE FINAL ANSWER (Streaming)
        # ==========================================
        yield {"step": "answer", "status": "start", "data": None}

        # Format lại context cuối cùng (bao gồm cả doc tìm được từ tool nếu có)
        final_context_text = self._format_context(final_context_docs)
        
        # Tạo map citation: {"1": doc_obj_1, "2": doc_obj_2}
        # Key là string index để khớp với format [1], [2] trong text
        citation_map = {str(i+1): d for i, d in enumerate(final_context_docs)}

        final_prompt = f"""Dựa trên ngữ cảnh sau:
{final_context_text}

Câu hỏi: {question}

Hãy trả lời câu hỏi trên. 
QUY TẮC TRÍCH DẪN BẮT BUỘC:
- Mọi thông tin lấy từ ngữ cảnh đều phải trích dẫn nguồn.
- Sử dụng CHÍNH XÁC định dạng [N] (ví dụ: [1], [2], [3]). 
- KHÔNG thêm khoảng trắng (không dùng [ 1 ]), KHÔNG dùng định dạng khác.
- Đặt mã trích dẫn ở cuối câu hoặc cuối ý tương ứng.
- Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ là không có thông tin. /no_think
"""

        # Reset messages cho lần trả lời cuối cùng để tránh nhiễu từ quá trình decision
        # Hoặc giữ nguyên history nếu muốn model nhớ bối cảnh trước đó. 
        # Ở đây ta dùng prompt mới hoàn toàn để đảm bảo tập trung vào context cuối cùng.
        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ]
        
        full_answer = ""
        
        try:
            # Gọi LLM lần cuối để sinh câu trả lời (Có Stream)
            for chunk in self.chat_service.generate_response(final_messages, stream=stream):
                if stream:
                    delta = ""
                    # Xử lý nhiều định dạng response khác nhau từ OpenAI client
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content or ""
                    elif hasattr(chunk, 'content'):
                        delta = chunk.content or ""
                    elif isinstance(chunk, dict):
                        delta = chunk.get('content', '') or chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    
                    if delta:
                        full_answer += delta
                        yield {
                            "step": "answer", 
                            "status": "streaming", 
                            "data": {
                                "chunk": delta,
                                # Gửi citation_map để frontend có thể highlight nếu cần
                                # Lưu ý: gửi toàn bộ map mỗi lần có thể nặng, tùy frontend xử lý
                                "citations": citation_map 
                            }
                        }
                else:
                    # Non-stream mode
                    if hasattr(chunk, 'content'):
                        full_answer = chunk.content
                    elif isinstance(chunk, dict):
                        full_answer = chunk.get('content', '')
                    else:
                        full_answer = str(chunk)
            
            yield {
                "step": "answer", 
                "status": "done", 
                "data": {
                    "text": full_answer,
                    "citations": citation_map,
                    "sources": final_context_docs # Gửi full source để debug hoặc hiển thị chi tiết
                }
            }
            
        except Exception as e:
            yield {
                "step": "answer", 
                "status": "error", 
                "data": {"error": str(e)}
            }