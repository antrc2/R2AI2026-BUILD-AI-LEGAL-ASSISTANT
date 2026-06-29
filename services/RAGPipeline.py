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
                "Sử dụng khi câu hỏi yêu cầu chi tiết từ một văn bản cụ thể đã được nhắc đến hoặc trích dẫn gián tiếp. "
                "Luôn cố gắng cung cấp 'content_query' để mô tả nội dung cần tìm. "
                "Các tham số 'dieu_filter' và 'khoan_filter' là tùy chọn để thu hẹp phạm vi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ref": {
                        "type": "string",
                        "description": (
                            "Số hiệu văn bản pháp luật ĐẦY ĐỦ (ví dụ: '36/2015/QĐ-TTg') hoặc tên đầy đủ. "
                            "KHÔNG điền số thứ tự trích dẫn dạng [1], [2]."
                        ),
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

class RAGPipeline:
    def __init__(self):
        self.chat_service = ChatService()
        self.search_service = SearchService()

    def _parse_sub_queries(self, content: str) -> List[str]:
        """Parse JSON response từ LLM để lấy danh sách sub-queries."""
        try:
            clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()
            data = json.loads(clean_content)
            queries = data.get("queries", [])
            if isinstance(queries, list) and len(queries) > 0:
                return queries
            if isinstance(data, list):
                return data
            return [content]
        except json.JSONDecodeError:
            return [content]

    def process(self, question: str, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Pipeline xử lý chính."""
        
        all_context_docs = [] 
        current_messages = [
            {"role": "system", "content": "Bạn là trợ lý pháp lý thông minh. Hãy trả lời chính xác dựa trên ngữ cảnh."},
            {"role": "user", "content": question}
        ]

        # --- BƯỚC 1: SUB-QUERY (JSON Format) ---
        sub_query_prompt = (
            f"Hãy phân tích câu hỏi sau thành các ý nhỏ (sub-queries) để tìm kiếm thông tin hiệu quả hơn. "
            f"Trả về kết quả dưới dạng JSON thuần túy với key 'queries' chứa danh sách các chuỗi. "
            f"Ví dụ: {{\"queries\": [\"ý 1\", \"ý 2\"]}}.\n\nCâu hỏi: {question}"
        )
        
        sub_query_messages = current_messages + [{"role": "user", "content": sub_query_prompt}]
        yield {"step": "sub_queries", "status": "processing", "data": None}

        sub_query_response = ""
        # Gọi LLM không stream để đảm bảo lấy trọn vẹn JSON
        for chunk in self.chat_service.generate_response(
            sub_query_messages, 
            response_format={"type": "json_object"}, 
            stream=False
        ):
            if hasattr(chunk, 'content'):
                sub_query_response = chunk.content
            elif isinstance(chunk, dict) and 'content' in chunk:
                sub_query_response = chunk['content']
        
        sub_queries = self._parse_sub_queries(sub_query_response)
        yield {"step": "sub_queries", "status": "done", "data": {"queries": sub_queries}}

        # --- BƯỚC 2: SEARCH (Semantic Search) ---
        yield {"step": "retrieval", "status": "processing", "data": None}
        
        retrieved_docs = []
        for sq in sub_queries:
            docs = self.search_service.semantic_search(query=sq, top_k=5)
            retrieved_docs.extend(docs)
        
        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            doc_id = doc.get('chunk_id', str(doc))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        initial_context = unique_docs[:10]
        all_context_docs.extend(initial_context)
        
        yield {
            "step": "retrieval", 
            "status": "done", 
            "data": {
                "count": len(initial_context),
                "results": [{"id": i, "snippet": d.get('content', '')[:100]} for i, d in enumerate(initial_context)]
            }
        }

        # --- BƯỚC 3: TOOL CALL DECISION ---
        context_text = "\n\n".join([f"[{i+1}]: {d.get('content')}" for i, d in enumerate(initial_context)])
        decision_prompt = (
            f"Dựa trên các tài liệu tìm được sau:\n{context_text}\n\n"
            f"Câu hỏi gốc: {question}\n\n"
            f"Hãy xem xét xem thông tin đã đủ để trả lời chưa. "
            f"Nếu trong tài liệu có nhắc đến một văn bản khác (ví dụ: 'được hướng dẫn tại Thông tư X', 'sửa đổi bởi Luật Y') "
            f"mà bạn cần chi tiết từ văn bản đó để trả lời chính xác, hãy gọi hàm 'search_referenced_document'. "
            f"Nếu đã đủ thông tin, hãy trả lời trực tiếp mà không cần gọi tool."
        )
        
        tool_messages = current_messages + [{"role": "user", "content": decision_prompt}]
        yield {"step": "tool_call", "status": "processing", "data": None}

        tool_calls_needed = []
        use_tool = False

        response_iter = self.chat_service.generate_response(
            tool_messages,
            tools=SEARCH_TOOLS,
            stream=False
        )
        
        message_result = next(response_iter)
        
        if hasattr(message_result, 'tool_calls') and message_result.tool_calls:
            use_tool = True
            for tc in message_result.tool_calls:
                if tc.function.name == "search_referenced_document":
                    args = json.loads(tc.function.arguments)
                    tool_calls_needed.append(args)
                    yield {"step": "tool_call", "status": "detected", "data": {"tool": "search_referenced_document", "args": args}}
                    
                    # Thực hiện tìm kiếm bổ sung
                    extra_docs = self.search_service.doc_ref_search(
                        query=args.get("content_query", question),
                        doc_ref=args.get("doc_ref"),
                        article_filter=args.get("dieu_filter"),
                        clause_filter=args.get("khoan_filter"),
                        top_k=5
                    )
                    all_context_docs.extend(extra_docs)
                    yield {"step": "tool_call", "status": "executed", "data": {"found_count": len(extra_docs)}}
        
        # Chuẩn bị context cuối cùng
        if use_tool:
            seen = set()
            final_context_list = []
            for d in all_context_docs:
                doc_id = d.get('chunk_id', str(d))
                if doc_id not in seen:
                    seen.add(doc_id)
                    final_context_list.append(d)
            
            context_text = "\n\n".join([f"[{i+1}]: {d.get('content')}" for i, d in enumerate(final_context_list)])
            final_prompt = (
                f"Dựa trên các tài liệu sau (đã bao gồm cả kết quả tìm kiếm bổ sung):\n{context_text}\n\n"
                f"Hãy trả lời câu hỏi: {question}\n\n"
                f"Yêu cầu: Trích dẫn nguồn rõ ràng bằng cú pháp [N] với N là số thứ tự tài liệu trong danh sách trên."
            )
            current_messages = [
                {"role": "system", "content": "Bạn là trợ lý pháp lý. Trả lời dựa trên ngữ cảnh provided. Dùng citation [N]."},
                {"role": "user", "content": final_prompt}
            ]
        else:
            yield {"step": "tool_call", "status": "skipped", "data": {"reason": "Đủ thông tin"}}
            context_text = "\n\n".join([f"[{i+1}]: {d.get('content')}" for i, d in enumerate(initial_context)])
            final_prompt = (
                f"Dựa trên các tài liệu sau:\n{context_text}\n\n"
                f"Hãy trả lời câu hỏi: {question}\n\n"
                f"Yêu cầu: Trích dẫn nguồn rõ ràng bằng cú pháp [N]."
            )
            current_messages = [
                {"role": "system", "content": "Bạn là trợ lý pháp lý. Trả lời dựa trên ngữ cảnh provided. Dùng citation [N]."},
                {"role": "user", "content": final_prompt}
            ]

        # --- BƯỚC 4: GENERATE ANSWER ---
        yield {"step": "answer", "status": "start", "data": None}

        citation_map = {str(i+1): d for i, d in enumerate(all_context_docs)}
        full_answer = ""
        
        for chunk in self.chat_service.generate_response(current_messages, stream=stream):
            if stream:
                delta = ""
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content or ""
                
                if delta:
                    full_answer += delta
                    yield {
                        "step": "answer", 
                        "status": "streaming", 
                        "data": {
                            "chunk": delta,
                            "citations": citation_map
                        }
                    }
            else:
                if hasattr(chunk, 'content'):
                    full_answer = chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    full_answer = chunk['content']
        
        yield {
            "step": "answer", 
            "status": "done", 
            "data": {
                "text": full_answer,
                "citations": citation_map,
                "sources": all_context_docs
            }
        }