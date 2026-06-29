"""
services/RAGPipeline.py - Pipeline RAG cho pháp luật Việt Nam
Refactored từ main_only_search.py để expose thành API
"""

import os
import json
import re
import time
from typing import Any, Generator, Optional, Dict, List

import faiss

# Import từ search_v2
from search_v2 import (
    load_index,
    load_faiss_id_map,
    load_chunk_map,
    load_article_index_map,
    faiss_search,
    rerank,
    search_by_doc_ref,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

URL_LLM_API = "http://10.9.3.75:31028/api/llama3/8b"
MODEL_LLM   = "llm3.1-sea"

RETRY_ATTEMPTS = 5
RETRY_DELAY    = 0.5

# Retrieval tuning knobs
PER_QUERY_FAISS_K  = 45
GLOBAL_RERANK_CAP  = 200
FINAL_CONTEXT_K    = 20
RERANK_THRESHOLD   = 0.35

ENABLE_FOLLOWUP_LOOKUP = False
MAX_FOLLOWUP_ROUNDS    = 2
FOLLOWUP_TOPK          = 6

# ─────────────────────────────────────────────────────────────
# LLM Client (icllmlib)
# ─────────────────────────────────────────────────────────────

try:
    from icllmlib import LLM
    llm = LLM(
        app_code="",
        function_code="",
        model_llm=MODEL_LLM,
        url_prompt="",
        llm_name="",
        url_llm_api=URL_LLM_API,
        prompt="",
        is_show_console=False,
        is_log=False,
        is_get_prompt_online=False
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm = None


class LLMClientError(RuntimeError):
    pass


class LLMCallError(LLMClientError):
    pass


def _call_llm(prompt: str, system_prompt: str, max_decoding_length: int = 2048,
              temperature: float = 0.1,
              retries: int = RETRY_ATTEMPTS, delay: float = RETRY_DELAY) -> str:
    """Gọi icllmlib.LLM.generate() và trả về answer_norm (string)."""
    if not LLM_AVAILABLE:
        raise LLMCallError("LLM client không khả dụng (icllmlib chưa được cài đặt)")
    
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = llm.generate(
                [],
                prompt=prompt,
                system_prompt=system_prompt,
                lang_output="vi",
                is_check_valid=False,
                is_translate_context=False,
                is_translate_prompt=False,
                is_translate_result=False,
                temperature=temperature,
                max_input_length=32000,
                max_decoding_length=max_decoding_length,
            )

            if not response:
                raise LLMCallError("LLM trả về response rỗng.")

            answer = (response[0].get("answer_norm") or response[0].get("answer") or "").strip()

            if not answer:
                raise LLMCallError("LLM trả về answer rỗng.")

            return answer

        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay)
            else:
                raise LLMCallError(f"Gọi LLM thất bại sau {retries} lần: {last_error}")
    
    raise LLMCallError(f"Gọi LLM thất bại: {last_error}")


JSON_FORMAT_RULES = """
QUY TẮC BẮT BUỘC VỀ FORMAT JSON:
- Chỉ trả về DUY NHẤT một JSON object hợp lệ, không kèm bất kỳ văn bản nào khác.
- Không dùng markdown code block (không có ```json hoặc ```).
- Toàn bộ JSON phải nằm trên một khối liền mạch, mở bằng { và đóng bằng }.
"""


def _parse_json_answer(answer: str) -> dict:
    normalized = answer.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*```$", "", normalized)

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
        if not match:
            raise LLMClientError("LLM response was not valid JSON.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON response must be an object.")
    return parsed


def _call_llm_json(prompt: str, system_prompt: str, max_decoding_length: int = 1024,
                   temperature: float = 0.1,
                   retries: int = RETRY_ATTEMPTS, delay: float = RETRY_DELAY) -> dict:
    """Gọi LLM và parse JSON."""
    full_system_prompt = f"{system_prompt}\n{JSON_FORMAT_RULES}"
    last_error = None
    
    for attempt in range(1, retries + 1):
        try:
            raw = _call_llm(prompt, full_system_prompt, max_decoding_length=max_decoding_length, temperature=temperature, retries=1)
            return _parse_json_answer(raw)
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(delay)
            else:
                raise LLMClientError(f"Gọi/Parse JSON thất bại: {last_error}")
    
    raise LLMClientError(f"Gọi/Parse JSON thất bại: {last_error}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _preview(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def parse_year_from_doc_num(doc_num: str) -> int:
    if not doc_num:
        return 0
    m = re.search(r'/(\d{4})/', doc_num)
    if m:
        return int(m.group(1))
    m = re.search(r'\b(19|20)\d{2}\b', doc_num)
    if m:
        return int(m.group(0))
    return 0


DOC_TYPE_PATTERNS: list = [
    (r"TTLT[-]", "Thông tư liên tịch"),
    (r"PL[-]?UBTVQH", "Pháp lệnh"),
    (r"NQ[-]?(CP|TW|QH)", "Nghị quyết"),
    (r"NĐ[-]?CP", "Nghị định"),
    (r"QĐ[-]?TTg", "Quyết định"),
    (r"QĐ[-]", "Quyết định"),
    (r"CT[-]?TTg", "Chỉ thị"),
    (r"TT[-]", "Thông tư"),
    (r"/QH\d*", "Luật"),
]


def guess_doc_type(doc_num: str) -> str:
    if not doc_num:
        return ""
    for pattern, label in DOC_TYPE_PATTERNS:
        if re.search(pattern, doc_num, flags=re.IGNORECASE | re.UNICODE):
            return label
    return ""


def build_doc_name(meta: dict) -> str:
    title = (meta.get("title") or "").strip()
    doc_num = (meta.get("doc_num") or "").strip()

    if not title:
        return doc_num

    if doc_num and doc_num in title:
        return title

    doc_type = guess_doc_type(doc_num)
    prefix = " ".join(p for p in (doc_type, doc_num) if p)
    return f"{prefix} {title}".strip() if prefix else title


def build_chunk_text(candidate: dict, chunk_map: dict) -> str:
    meta = candidate.get("metadata", {})
    parts = []
    if meta.get("title"):
        parts.append(meta["title"])
    art = meta.get("article") or meta.get("dieu_so") or ""
    if art:
        parts.append(art)
    lt = meta.get("leaf_title", "")
    if lt and lt != art:
        parts.append(lt)
    return " > ".join(parts)


def build_context_with_index(
    search_results: list,
    chunks_text_map: dict,
) -> tuple:
    ref_index = []
    blocks    = []

    for i, res in enumerate(search_results, start=1):
        meta     = res.get("metadata", {})
        chunk_id = res.get("chunk_id", "")
        doc_num  = (meta.get("doc_num")  or "").strip()
        doc_name = build_doc_name(meta)
        dieu_so  = (meta.get("dieu_so")  or "").strip()
        article  = (meta.get("article")  or dieu_so).strip()
        year     = parse_year_from_doc_num(doc_num)
        text     = chunks_text_map.get(chunk_id, build_chunk_text(res, {}))

        ref_index.append({
            "ref_id":   i,
            "doc_num":  doc_num,
            "doc_name": doc_name,
            "article":  article,
            "dieu_so":  dieu_so,
            "year":     year,
            "text":     text,
            "chunk_id": chunk_id,
            "title": meta.get('title')
        })

        header = f"[{i}] {doc_name}"
        if article:
            header += f" | {article}"
        if year:
            header += f" (năm {year})"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks), ref_index


_CITATION_PATTERN = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')


def extract_used_refs_from_answer(answer: str, max_ref_id: int) -> list:
    """Trích xuất các số [N] từ câu trả lời."""
    used = []
    seen = set()
    for m in _CITATION_PATTERN.finditer(answer or ""):
        for part in m.group(1).split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            n = int(part)
            if 1 <= n <= max_ref_id and n not in seen:
                seen.add(n)
                used.append(n)
    return used


def build_relevant_from_used_refs(used_refs: list, ref_index: list) -> tuple:
    seen_docs     = {}
    seen_articles = {}

    for ref_id in used_refs:
        idx = ref_id - 1
        if idx < 0 or idx >= len(ref_index):
            continue

        entry    = ref_index[idx]
        doc_num  = entry["doc_num"]
        doc_name = entry["doc_name"]
        dieu_so  = entry.get("dieu_so", "") or entry.get("article", "")

        if not doc_num:
            continue

        if doc_num not in seen_docs:
            seen_docs[doc_num] = doc_name

        if dieu_so:
            art_key = f"{doc_num}|{dieu_so}"
            if art_key not in seen_articles:
                seen_articles[art_key] = f"{doc_num}|{doc_name}|{dieu_so}"

    relevant_docs     = [f"{dn}|{nm}" for dn, nm in seen_docs.items()]
    relevant_articles = list(seen_articles.values())
    return relevant_docs, relevant_articles


# ─────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────

def llm_sub_queries(question: str) -> list:
    system_prompt = "Trả về valid JSON object, không markdown, không giải thích thêm."

    prompt = f"""Bạn là chuyên gia phân tích cú pháp câu hỏi pháp luật Việt Nam.

Câu hỏi gốc: "{question}"

Nhiệm vụ: PHÂN RÃ (decompose) câu hỏi gốc thành các câu hỏi con.

Trả về JSON object:
{{"sub_queries": ["câu hỏi con 1", "câu hỏi con 2"]}}"""

    try:
        data = _call_llm_json(prompt, system_prompt, max_decoding_length=1024)
        sqs = data.get("sub_queries", [])
        return sqs if sqs else [question]
    except Exception:
        return [question]


ANSWER_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp luật Việt Nam, trả lời cho người dùng phổ thông.

NGUYÊN TẮC NỘI DUNG (bắt buộc):
1. CHỈ sử dụng thông tin có trong TÀI LIỆU THAM KHẢO được cung cấp.
2. TRƯỚC KHI trích dẫn một điều khoản, kiểm tra đối tượng/phạm vi áp dụng.
3. "Đầy đủ và toàn diện" nghĩa là trả lời HẾT các khía cạnh THỰC SỰ được hỏi.

NGUYÊN TẮC TRÍCH DẪN (bắt buộc):
- Mỗi khi nêu một quy định, PHẢI nêu rõ TRONG CÙNG CÂU: loại văn bản + số hiệu văn bản + số điều, rồi đặt số thứ tự tài liệu trong dấu vuông [N].
- Định dạng số tài liệu: [N] cho một tài liệu, hoặc [N, M] khi một câu dựa trên nhiều tài liệu cùng lúc.

NGUYÊN TẮC HÌNH THỨC:
- Câu đầu tiên: trả lời TRỰC TIẾP và ngắn gọn vào trọng tâm câu hỏi.
- Dùng tiếng Việt rõ ràng."""


def llm_answer(question: str, context_text: str, ref_index: list) -> tuple:
    """Sinh câu trả lời và trích xuất used_refs."""
    
    base_prompt = f"""TÀI LIỆU THAM KHẢO (đánh số để trích dẫn):
{context_text}

---

CÂU HỎI: {question}

Hãy trả lời dựa trên tài liệu, tuân thủ đúng các nguyên tắc trích dẫn ở trên."""

    try:
        answer = _call_llm(base_prompt, ANSWER_SYSTEM_PROMPT, max_decoding_length=3072)
    except Exception as e:
        answer = f"Lỗi khi sinh câu trả lời: {str(e)}"

    # Trích used_refs từ câu trả lời
    used_refs = extract_used_refs_from_answer(answer, max_ref_id=len(ref_index))

    # Retry nếu không tìm thấy ref nào
    if not used_refs and ref_index and answer:
        reminder_prompt = base_prompt + """

LƯU Ý QUAN TRỌNG: câu trả lời BẮT BUỘC phải đánh số [N] ngay sau mỗi nội dung được trích từ tài liệu tham khảo."""
        try:
            answer_retry = _call_llm(reminder_prompt, ANSWER_SYSTEM_PROMPT, max_decoding_length=3072)
            retry_refs = extract_used_refs_from_answer(answer_retry, max_ref_id=len(ref_index))
            if retry_refs:
                answer, used_refs = answer_retry, retry_refs
        except Exception:
            pass

    return answer, used_refs, ref_index


# ─────────────────────────────────────────────────────────────
# RAGPipeline Class
# ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """Pipeline RAG cho pháp luật Việt Nam."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index = None
        self.faiss_id_map = None
        self.chunk_map = None
        self.article_index_map = None
        self.chunks_text_map = {}
        self._initialized = False
    
    def initialize(self):
        """Khởi tạo pipeline (load index, maps)."""
        if self._initialized:
            return
        
        # Set paths
        global FAISS_INDEX_FILE, FAISS_ID_MAP_FILE, CHUNK_MAP_FILE, ARTICLE_MAP_FILE
        FAISS_INDEX_FILE   = os.path.join(self.data_dir, "faiss.index")
        FAISS_ID_MAP_FILE  = os.path.join(self.data_dir, "faiss_id_map.json")
        CHUNK_MAP_FILE     = os.path.join(self.data_dir, "chunk_map.json")
        ARTICLE_MAP_FILE   = os.path.join(self.data_dir, "article_index_map.json")
        CHUNKS_FILE        = os.path.join(self.data_dir, "chunks.json")
        
        self.index = load_index()
        self.faiss_id_map = load_faiss_id_map()
        self.chunk_map = load_chunk_map()
        self.article_index_map = load_article_index_map()
        
        # Load chunks text map
        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            for c in all_chunks:
                self.chunks_text_map[c["chunk_id"]] = c.get("embed_text", "")
        
        self._initialized = True
    
    def process(self, question: str, stream: bool = False) -> Generator[dict, None, None]:
        """
        Xử lý câu hỏi và trả về kết quả qua generator.
        
        Nếu stream=True: yield từng bước của pipeline.
        Nếu stream=False: yield một lần duy nhất với kết quả cuối cùng.
        """
        if not self._initialized:
            self.initialize()
        
        # Step 1: Sub-queries
        if stream:
            yield {
                "step": "sub_queries",
                "status": "processing",
                "message": "Đang phân tích câu hỏi thành sub-queries..."
            }
        
        sub_queries = llm_sub_queries(question)
        
        if stream:
            yield {
                "step": "sub_queries",
                "status": "completed",
                "data": {"sub_queries": sub_queries, "count": len(sub_queries)}
            }
        
        # Step 2: Retrieval (FAISS + Rerank)
        if stream:
            yield {
                "step": "retrieval",
                "status": "processing",
                "message": "Đang tìm kiếm tài liệu..."
            }
        
        pool = []
        seen_chunk_ids = set()
        
        for sq_idx, sq in enumerate([question] + sub_queries):
            raw_results = faiss_search(
                query=sq,
                index=self.index,
                faiss_id_map=self.faiss_id_map,
                chunk_map=self.chunk_map,
                top_k=PER_QUERY_FAISS_K,
            )
            for res in raw_results:
                cid = res.get("chunk_id", "")
                if cid and cid in seen_chunk_ids:
                    continue
                if cid:
                    seen_chunk_ids.add(cid)
                pool.append(res)
        
        if len(pool) > GLOBAL_RERANK_CAP:
            pool.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            pool = pool[:GLOBAL_RERANK_CAP]
        
        reranked = rerank(question, pool, top_k=FINAL_CONTEXT_K)
        all_candidates = [c for c in reranked if c.get("rerank_score", 0.0) >= RERANK_THRESHOLD]
        
        if stream:
            # Chuẩn hóa thông tin retrieval để trả về
            retrieval_results = []
            for c in all_candidates[:10]:  # Giới hạn 10 kết quả để hiển thị
                meta = c.get("metadata", {})
                retrieval_results.append({
                    "ref_id": len(retrieval_results) + 1,
                    "doc_num": meta.get("doc_num", ""),
                    "doc_name": build_doc_name(meta),
                    "article": meta.get("article", "") or meta.get("dieu_so", ""),
                    "score": c.get("rerank_score", 0.0),
                    "text": self.chunks_text_map.get(c.get("chunk_id", ""), "")[:200] + "..."
                })
            
            yield {
                "step": "retrieval",
                "status": "completed",
                "data": {
                    "total_found": len(all_candidates),
                    "results": retrieval_results
                }
            }
        
        # Build context
        if all_candidates:
            context_text, ref_index = build_context_with_index(all_candidates, self.chunks_text_map)
        else:
            context_text = "Không tìm thấy tài liệu liên quan đủ độ tin cậy."
            ref_index = []
        
        # Step 3: Tool call check (optional - hiện tại không dùng tool)
        if stream:
            yield {
                "step": "tool_call",
                "status": "completed",
                "data": {"used_tool": False, "message": "Không sử dụng tool ngoài"}
            }
        
        # Step 4: Generate answer
        if stream:
            yield {
                "step": "answer",
                "status": "processing",
                "message": "Đang sinh câu trả lời..."
            }
        
        answer, used_refs, ref_index = llm_answer(question, context_text, ref_index)
        
        # Build relevant docs/articles
        relevant_docs, relevant_articles = build_relevant_from_used_refs(used_refs, ref_index)
        
        # Build references info for hover
        references_info = []
        for ref_id in used_refs:
            idx = ref_id - 1
            if 0 <= idx < len(ref_index):
                entry = ref_index[idx]
                references_info.append({
                    "ref_id": ref_id,
                    "doc_num": entry["doc_num"],
                    "doc_name": entry["doc_name"],
                    "article": entry["article"],
                    "text": entry["text"]
                })
        
        if stream:
            yield {
                "step": "answer",
                "status": "completed",
                "data": {
                    "answer": answer,
                    "used_refs": used_refs,
                    "relevant_docs": relevant_docs,
                    "relevant_articles": relevant_articles,
                    "references_info": references_info
                }
            }
        
        # Final result
        if stream:
            yield {
                "step": "final",
                "status": "completed",
                "data": {
                    "answer": answer,
                    "used_refs": used_refs,
                    "relevant_docs": relevant_docs,
                    "relevant_articles": relevant_articles,
                    "references_info": references_info
                }
            }
        else:
            # Non-stream: chỉ yield kết quả cuối cùng
            yield {
                "step": "final",
                "status": "completed",
                "data": {
                    "answer": answer,
                    "used_refs": used_refs,
                    "relevant_docs": relevant_docs,
                    "relevant_articles": relevant_articles,
                    "references_info": references_info,
                    "pipeline_steps": {
                        "sub_queries": sub_queries,
                        "retrieval_count": len(all_candidates)
                    }
                }
            }


# Singleton instance
_pipeline_instance = None


def get_pipeline(data_dir: str = "data") -> RAGPipeline:
    """Get singleton pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(data_dir=data_dir)
    return _pipeline_instance
