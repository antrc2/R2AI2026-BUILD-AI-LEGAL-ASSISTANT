"""
main_v2.py – Pipeline RAG pháp luật Việt Nam (schema mới, host bằng LM Studio)
"""

import os
import json
import re
import faiss
from openai import OpenAI

from search_v2 import (
    load_index,
    load_faiss_id_map,
    load_chunk_map,
    search_and_rerank,
    search_by_doc_ref,
    load_article_index_map,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CHAT_URL   = "http://localhost:11111/v1"
CHAT_MODEL = "Qwen3-14B-Q8_0.gguf"

INPUT_FILE  = "R2AIStage1DATA.json"
OUTPUT_FILE = "results.json"

chat_client = OpenAI(base_url=CHAT_URL, api_key="string")

# extra_body dùng để tắt "thinking" (chế độ suy luận) khi model hỗ trợ
# chat_template_kwargs (Qwen3 trên LM Studio). Áp dụng cho MỌI lệnh gọi LLM.
NO_THINK_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# Số lần retry tối đa cho mỗi giai đoạn LLM khi gặp lỗi (gọi API lỗi, hoặc
# model không tuân thủ format yêu cầu — ví dụ không gọi tool khi bị ép).
MAX_RETRIES = 3

# Pattern phát hiện model nhầm doc_ref với ref_id dạng [N] (chỉ là số nguyên,
# có hoặc không có ngoặc vuông) — doc_ref hợp lệ luôn chứa chữ hoặc dấu '/'.
_REF_ID_MISTAKE_PATTERN = re.compile(r'^\[?\d+\]?$')


# ─────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────

def _sep(char="─", n=60):
    print(char * n)

def _step(num: int, label: str):
    print(f"\n  ┌─ Bước {num}: {label}")

def _done(label: str, value: str = ""):
    suffix = f" → {value}" if value else ""
    print(f"  └─ ✓ {label}{suffix}")

def _info(msg: str):
    print(f"  │  {msg}")

def _warn(msg: str):
    print(f"  │  ⚠ {msg}")

def _tool(msg: str):
    print(f"  │  🔧 {msg}")

def _preview(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ─────────────────────────────────────────────────────────────
# Helpers – metadata & context
# ─────────────────────────────────────────────────────────────

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


def build_doc_name(meta: dict) -> str:
    # FIX (port từ v1): dùng .get() để tránh KeyError/AttributeError khi
    # thiếu 'title' hoặc title=None.
    return (meta.get("title") or "").strip()


def build_chunk_text(candidate: dict, chunk_map: dict) -> str:
    meta = candidate.get("metadata", {})
    parts = []
    if meta.get("title"):
        parts.append(meta["title"])

    # FIX (port từ v1): ưu tiên lấy dieu_so, fallback sang article
    art = meta.get("article") or meta.get("dieu_so") or ""
    if art:
        parts.append(art)

    lt = meta.get("leaf_title", "")
    if lt and lt != art:
        parts.append(lt)
    return " > ".join(parts)


def build_context_with_index(
    search_results: list[dict],
    chunks_text_map: dict,
) -> tuple[str, list[dict]]:
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
            "title":    meta.get("title"),
        })

        header = f"[{i}] {doc_name}"
        if article:
            header += f" | {article}"
        if year:
            header += f" (năm {year})"
        if doc_num:
            header += f" | số hiệu: {doc_num}"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks), ref_index


def build_relevant_from_used_refs(
    used_refs: list[int],
    ref_index: list[dict],
) -> tuple[list[str], list[str]]:
    seen_docs     = {}
    seen_articles = {}

    for ref_id in used_refs:
        idx = ref_id - 1
        if idx < 0 or idx >= len(ref_index):
            _warn(f"ref_id={ref_id} ngoài phạm vi ref_index, bỏ qua.")
            continue

        entry    = ref_index[idx]
        doc_num  = entry["doc_num"]
        doc_name = entry["doc_name"]
        dieu_so  = entry.get("dieu_so", "") or entry.get("article", "")

        if not doc_num:
            _warn(f"ref_id={ref_id} không có doc_num, bỏ qua.")
            continue

        if doc_num not in seen_docs:
            seen_docs[doc_num] = doc_name

        if dieu_so:
            art_key = f"{doc_num}|{dieu_so}"
            if art_key not in seen_articles:
                seen_articles[art_key] = f"{doc_num}|{doc_name}|{dieu_so}"
        else:
            _warn(f"ref_id={ref_id} không có dieu_so – chỉ vào relevant_docs.")

    relevant_docs     = [f"{dn}|{nm}" for dn, nm in seen_docs.items()]
    relevant_articles = list(seen_articles.values())
    return relevant_docs, relevant_articles


# ─────────────────────────────────────────────────────────────
# Function calling tools definition
# ─────────────────────────────────────────────────────────────

# Tool dùng riêng cho Giai đoạn 1 (sinh sub-queries). LLM bị ép (tool_choice
# = required tool này) phải gọi tool để trả kết quả có cấu trúc, thay vì viết
# JSON tự do trong content — tránh việc model nhả nhầm format ở giai đoạn sau.


SUB_QUERIES_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_sub_queries",
            "description": (
                "Nộp danh sách sub-query đã phân rã (decompose) từ câu hỏi gốc. "
                "Luôn gọi tool này để trả kết quả, không trả lời bằng văn bản thường."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Danh sách câu hỏi con, mỗi câu là một câu hoàn chỉnh bằng tiếng Việt. "
                            "Thường chỉ 1-3 câu."
                        ),
                    },
                },
                "required": ["sub_queries"],
            },
        },
    }
]


SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_referenced_document",
            "description": (
                "Tìm kiếm nội dung cụ thể trong một văn bản pháp luật được trích dẫn. "
                "Luôn cố gắng cung cấp 'dieu_filter' và/hoặc 'khoan_filter' hoặc 'content_query' "
                "để thu hẹp phạm vi tìm kiếm, tránh việc trả về toàn bộ văn bản gây tràn bộ nhớ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ref": {
                        "type": "string",
                        "description": (
                            "Số hiệu văn bản pháp luật ĐẦY ĐỦ (ví dụ: '36/2015/QĐ-TTg', "
                            "'04/2017/QH14') hoặc tên đầy đủ của văn bản (Bắt buộc). "
                            "TUYỆT ĐỐI KHÔNG điền số thứ tự trích dẫn dạng [1], [2], hay chỉ "
                            "một số nguyên đơn lẻ — đó là ref_id để trích dẫn, KHÔNG phải doc_ref. "
                            "Lấy đúng số hiệu/tên thật xuất hiện trong tài liệu tham khảo, ví dụ "
                            "nếu tài liệu ghi '[3] Luật ABC | Điều 13 | số hiệu: 04/2017/QH14' "
                            "thì doc_ref = '04/2017/QH14' (hoặc 'Luật ABC' nếu không thấy số hiệu)."
                        ),
                    },
                    "dieu_filter": {
                        "type": "string",
                        "description": "(Khuyến nghị) Chỉ ghi số điều, ví dụ 'Điều 74'. Để rỗng nếu không xác định được điều cụ thể.",
                    },
                    "khoan_filter": {
                        "type": "string",
                        "description": "(Khuyến nghị) Chỉ ghi số khoản, ví dụ 'Khoản 3'. Để rỗng nếu không có hoặc cần cả điều. KHÔNG gộp điều và khoản vào cùng một field.",
                    },
                    "content_query": {
                        "type": "string",
                        "description": "(Khuyến nghị) Từ khóa hoặc chủ đề cần tìm trong văn bản đó dựa trên ngữ cảnh câu hỏi.",
                    },
                },
                "required": ["doc_ref"],
            },
        },
    }
]

# response_format (structured output) dùng riêng cho Giai đoạn 3 (used_refs).
# Ép model trả đúng schema {"used_refs": [int, ...]} thay vì JSON tự do.
USED_REFS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "used_refs_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "used_refs": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["used_refs"],
            "additionalProperties": False,
        },
    },
}


# ─────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────

def llm_full_pipeline(
    question: str,
    index: faiss.Index,
    faiss_id_map: dict,
    chunk_map: dict,
    article_index_map: dict,
    chunks_text_map: dict,
    search_fn,
    max_tool_rounds: int = 3,
) -> tuple[list[str], str, list[int], list[dict], str]:
    """
    Toàn bộ pipeline LLM cho 1 câu hỏi chạy trong CÙNG MỘT conversation
    (messages list duy nhất, nối tiếp nhau), gồm 3 giai đoạn:

      1) Sinh sub-queries: ÉP LLM gọi tool `submit_sub_queries` (tool_choice
         = required tool này), KHÔNG dựa vào content JSON tự do. Retry tối
         đa MAX_RETRIES lần nếu model không gọi tool; hết retry → fallback
         dùng câu hỏi gốc làm sub-query duy nhất.

      2) Trả lời dựa trên context (sau khi đã search bằng sub-queries).
         Giai đoạn này CHỈ cho phép gọi tool `search_referenced_document`
         (tool_choice="auto" — model tự quyết định có cần tra cứu thêm
         hay trả lời thẳng). Retry tối đa MAX_RETRIES lần nếu gọi API lỗi.
         Nếu model gọi tool không hợp lệ (sai tên, hoặc doc_ref bị nhầm
         với ref_id [N]), trả lỗi rõ ràng qua tool result để model tự sửa
         ở round sau, KHÔNG tính là lỗi gọi API.

      3) Khai báo used_refs: ép schema bằng response_format (structured
         output / json_schema), không tool, không parse JSON tự do từ
         content. Retry tối đa MAX_RETRIES lần nếu gọi API lỗi.

    `search_fn` là callback (sub_queries: list[str]) -> (context_text, ref_index),
    được gọi giữa giai đoạn 1 và 2 vì cần có sub-queries trước khi search FAISS.

    Trả về: (sub_queries, answer, used_refs, ref_index, context_text)
    """
    system_msg = """Bạn là chuyên gia tư vấn pháp luật Việt Nam, hỗ trợ qua 3 giai đoạn nối tiếp
trong cùng một cuộc hội thoại: (1) phân rã câu hỏi thành sub-query bằng tool, (2) trả lời dựa
trên tài liệu tham khảo (có thể dùng tool tra cứu thêm), (3) khai báo các tài liệu đã sử dụng.

Nguyên tắc trả lời (áp dụng cho giai đoạn 2):
- Chỉ sử dụng thông tin có trong tài liệu tham khảo
- TRƯỚC KHI trích dẫn một điều khoản, kiểm tra: đối tượng/phạm vi áp dụng của điều khoản đó
  có khớp với đối tượng/tình huống trong câu hỏi không. Nếu điều khoản nói về một đối tượng
  khác (ví dụ: doanh nghiệp trong khu công nghiệp, trong khi câu hỏi hỏi về doanh nghiệp nhỏ
  và vừa nói chung), KHÔNG trích dẫn điều khoản đó, dù chủ đề chung có vẻ liên quan.
- Trích dẫn rõ: tên văn bản, số hiệu, điều khoản, dùng ký hiệu [N]. Ví dụ [1] [2]
- Nếu tài liệu đề cập đến một văn bản khác chưa có trong ngữ cảnh, hãy dùng tool search_referenced_document để tra cứu thêm.
  Khi gọi tool này, doc_ref PHẢI là số hiệu văn bản thật (ví dụ '04/2017/QH14') hoặc tên văn bản,
  LẤY TỪ NỘI DUNG tài liệu tham khảo — KHÔNG được dùng số thứ tự trích dẫn [N] làm doc_ref.
- Ưu tiên văn bản có năm ban hành mới hơn khi mâu thuẫn
- Trả lời bằng văn bản thuần, không JSON, không markdown code block
- Nếu có 2 văn bản giống nhau, thì sử dụng văn bản mới nhất (dựa theo năm)"""

    messages = [{"role": "system", "content": system_msg}]

    # ═══════════════════════════════════════════════════════
    # Giai đoạn 1: Sub-queries — ÉP gọi tool submit_sub_queries
    # ═══════════════════════════════════════════════════════
    _step(1, "Sinh sub-queries (ép dùng tool, cùng conversation)")

    sub_queries_prompt = f"""Bạn là chuyên gia phân tích cú pháp câu hỏi pháp luật Việt Nam.

Câu hỏi gốc: "{question}"

Nhiệm vụ: PHÂN RÃ (decompose) câu hỏi gốc thành các câu hỏi con, KHÔNG phải tạo thêm ý mới.
Gọi tool `submit_sub_queries` để nộp kết quả — KHÔNG trả lời bằng văn bản thường.

Quy tắc bắt buộc:
- Mỗi sub-query CHỈ ĐƯỢC chứa thông tin/thực thể/điều kiện đã CÓ SẴN trong câu hỏi gốc.
- TUYỆT ĐỐI KHÔNG suy diễn ra chủ thể, hậu quả, hay khía cạnh pháp lý mà câu hỏi gốc không
  đề cập (ví dụ: câu gốc hỏi về "công ty nộp chậm tiền thuế" thì KHÔNG được tự suy ra câu hỏi
  về "trách nhiệm người đại diện pháp luật" hay "lãi suất chậm nộp" nếu câu gốc không hỏi điều đó).
- Nếu câu hỏi gốc CHỈ chứa một ý duy nhất (một hành vi, một điều kiện, một đối tượng) và không
  thể tách nhỏ hơn mà không làm mất hoặc bóp méo nghĩa, hãy nộp CHÍNH XÁC 1 sub-query là câu hỏi
  gốc được viết lại rõ ràng hơn (chuẩn hóa thuật ngữ pháp lý), không thêm nội dung mới.
- Nếu câu hỏi gốc ghép nhiều ý (ví dụ: vừa hỏi về hành vi vi phạm vừa hỏi về mức xử phạt vừa hỏi
  về thời hạn), hãy tách mỗi ý thành 1 sub-query riêng, mỗi câu là một CÂU HOÀN CHỈNH bằng tiếng Việt.
- Giữ nguyên core subject (chủ thể + hành vi chính) của câu gốc trong MỌI sub-query.
- Không thêm ví dụ minh họa, không thêm trường hợp giả định, không mở rộng phạm vi câu hỏi.
- Số lượng sub-query phụ thuộc vào số ý có thật trong câu hỏi gốc, KHÔNG cố tạo ra càng nhiều
  câu càng tốt. Thường chỉ 1-3 câu là đủ; chỉ tạo nhiều hơn nếu câu gốc thực sự ghép nhiều ý.

Ví dụ ĐÚNG:
Câu gốc: "Doanh nghiệp nhỏ và vừa được ưu đãi thuế thu nhập doanh nghiệp như thế nào, và thủ tục đăng ký hưởng ưu đãi ra sao?"
→ gọi submit_sub_queries(sub_queries=[
    "Doanh nghiệp nhỏ và vừa được ưu đãi về thuế thu nhập doanh nghiệp như thế nào",
    "Thủ tục đăng ký hưởng ưu đãi thuế thu nhập doanh nghiệp đối với doanh nghiệp nhỏ và vừa"
  ])

Ví dụ SAI (KHÔNG làm theo - đây là việc "tư vấn mở rộng" chứ không phải phân rã):
Câu gốc: "Công ty tự tính thuế nhưng nộp chậm hơn ngày cuối cùng của thời hạn nộp hồ sơ khai thuế thì bị xử phạt thế nào?"
→ Sai vì tự suy ra các câu KHÔNG có trong câu gốc như:
  - "Trách nhiệm pháp lý của người đại diện pháp luật khi doanh nghiệp nộp thuế chậm hạn" (KHÔNG được hỏi)
  - "Lãi suất chậm nộp tiền thuế áp dụng khi doanh nghiệp nộp thuế sau hạn" (KHÔNG được hỏi, đây là suy diễn)
  - "Đối tượng chịu xử phạt trong trường hợp..." (KHÔNG được hỏi, đây là góc nhìn khác do LLM tự thêm)
Đúng ra với câu hỏi này chỉ nên nộp 1 sub-query (vì chỉ có 1 ý: nộp chậm thuế tự tính thì bị xử phạt gì):
→ gọi submit_sub_queries(sub_queries=["Công ty tự tính thuế nộp chậm so với thời hạn nộp hồ sơ khai thuế thì bị xử phạt vi phạm hành chính như thế nào"])

Hãy gọi tool submit_sub_queries ngay. /no_think"""

    messages.append({"role": "user", "content": sub_queries_prompt})

    sub_queries = [question]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            sq_resp = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=SUB_QUERIES_TOOLS,
                tool_choice="required",
                extra_body=NO_THINK_EXTRA_BODY,
            )
            sq_msg = sq_resp.choices[0].message

            if not sq_msg.tool_calls:
                # Model không gọi tool dù đã bị ép — append lại đúng những gì
                # model trả lời (giữ messages hợp lệ/xen kẽ role) rồi raise để
                # vào nhánh except và retry.
                messages.append({"role": "assistant", "content": sq_msg.content or ""})
                raise ValueError(
                    f"Model không gọi tool submit_sub_queries (content: {_preview(sq_msg.content or '', 100)!r})"
                )

            tc = sq_msg.tool_calls[0]
            args = json.loads(tc.function.arguments)
            sqs = args.get("sub_queries", [])

            if not sqs:
                # Vẫn append message tool_call (hợp lệ về cấu trúc) trước khi
                # raise, vì model ĐÃ gọi đúng tool — chỉ là payload rỗng.
                messages.append(sq_msg)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      "(lỗi: sub_queries rỗng)",
                })
                raise ValueError("Tool submit_sub_queries trả về sub_queries rỗng.")

            # Thành công: append đúng cặp assistant(tool_call) + tool(result)
            # để giữ messages hợp lệ cho các turn sau trong cùng conversation.
            messages.append(sq_msg)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps({"sub_queries": sqs}, ensure_ascii=False),
            })
            sub_queries = sqs
            _done("Sub-queries", f"{len(sub_queries)} queries (lần thử {attempt}/{MAX_RETRIES})")
            break

        except Exception as e:
            _warn(f"Lỗi sinh sub-queries (lần {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                messages.append({
                    "role": "user",
                    "content": (
                        "Bạn PHẢI gọi tool submit_sub_queries với tham số sub_queries là một danh "
                        "sách câu hỏi con. Không trả lời bằng văn bản thường. Hãy thử lại. /no_think"
                    ),
                })
            else:
                _warn("Hết số lần retry. Fallback dùng câu hỏi gốc làm sub-query duy nhất.")
                sub_queries = [question]
                messages.append({
                    "role": "user",
                    "content": "(Hệ thống tự động dùng câu hỏi gốc làm sub-query do không nhận được phản hồi hợp lệ.)",
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"sub_queries": sub_queries}, ensure_ascii=False),
                })

    for i, sq in enumerate(sub_queries, 1):
        _info(f"  [{i}] {_preview(sq, 100)}")

    # ── Search FAISS + build context (không phải lệnh gọi LLM) ──
    context_text, ref_index = search_fn(sub_queries)

    # ═══════════════════════════════════════════════════════
    # Giai đoạn 2: Answer — CHỈ cho phép tool search_referenced_document
    # ═══════════════════════════════════════════════════════
    _step(3, "LLM sinh câu trả lời (chỉ dùng tool search_referenced_document)")

    answer_prompt = f"""TÀI LIỆU THAM KHẢO (đánh số để trích dẫn):
{context_text}

---

CÂU HỎI: {question}

Hãy trả lời dựa trên tài liệu. Nếu tài liệu trích dẫn sang văn bản khác chưa có trong danh sách,
hãy dùng tool search_referenced_document để tìm thêm. Khi gọi tool, doc_ref PHẢI là số hiệu văn
bản thật (ví dụ '04/2017/QH14') hoặc tên văn bản lấy từ nội dung tài liệu — KHÔNG dùng số thứ tự
trích dẫn [N] làm doc_ref.

LƯU Ý: Câu trả lời cuối cùng PHẢI là văn bản thuần (plain text) bằng tiếng Việt, KHÔNG JSON,
KHÔNG markdown code block. /no_think"""

    messages.append({"role": "user", "content": answer_prompt})

    next_ref_id = len(ref_index) + 1
    final_answer_text = ""

    for round_idx in range(max_tool_rounds + 1):
        _info(f"Gọi LLM lần {round_idx + 1}/{max_tool_rounds + 1}...")

        resp = None
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = chat_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    tools=SEARCH_TOOLS,
                    tool_choice="auto",
                    max_tokens=2048,
                    extra_body=NO_THINK_EXTRA_BODY,
                )
                break
            except Exception as e:
                last_error = e
                _warn(f"Lỗi gọi LLM trả lời (lần {attempt}/{MAX_RETRIES}): {e}")

        if resp is None:
            _warn(f"Gọi LLM trả lời thất bại sau {MAX_RETRIES} lần: {last_error}. Dừng giai đoạn 2.")
            final_answer_text = ""
            break

        msg           = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason

        if finish_reason != "tool_calls" or not msg.tool_calls:
            final_answer_text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": final_answer_text})
            _done("LLM trả lời xong", f"{len(final_answer_text)} ký tự")
            _info(f"Preview: {_preview(final_answer_text, 150)}")
            break

        # Có thể có tool call hợp lệ (search_referenced_document) hoặc tool
        # call sai tên do model bị "dính" theo pattern giai đoạn 1
        # (submit_sub_queries) — cả hai đều được xử lý ở dưới, không raise.
        n_tools = len(msg.tool_calls)
        _info(f"LLM yêu cầu gọi {n_tools} tool(s)...")
        messages.append(msg)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            if fn_name == "search_referenced_document":
                doc_ref       = (args.get("doc_ref") or "").strip()
                dieu_filter   = args.get("dieu_filter") or None
                khoan_filter  = args.get("khoan_filter") or None
                content_query = args.get("content_query")

                # FIX: chặn sớm khi model nhầm doc_ref với ref_id dạng [N]
                # hoặc số nguyên thuần (vd: '1', '[2]') — đây không phải số
                # hiệu/tên văn bản hợp lệ. Trả lỗi rõ ràng qua tool result để
                # model tự sửa ở round sau, tránh lặp lại y nguyên lỗi cũ.
                if not doc_ref or _REF_ID_MISTAKE_PATTERN.match(doc_ref):
                    _warn(
                        f"doc_ref='{doc_ref}' không hợp lệ (nghi bị nhầm với ref_id [N])"
                    )
                    tool_result = (
                        f"LỖI: doc_ref='{doc_ref}' không hợp lệ. doc_ref KHÔNG phải là số "
                        f"thứ tự trích dẫn [N] — đó chỉ dùng để trích dẫn câu trả lời. "
                        f"doc_ref PHẢI là số hiệu văn bản thật (ví dụ '04/2017/QH14') hoặc "
                        f"tên văn bản, lấy từ dòng 'số hiệu: ...' hoặc tên văn bản tương ứng "
                        f"với [{doc_ref}] trong tài liệu tham khảo đã cho. Hãy gọi lại tool "
                        f"với doc_ref đúng."
                    )
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      tool_result,
                    })
                    continue

                _tool(f"search_referenced_document: '{doc_ref}'"
                      + (f" | {dieu_filter}" if dieu_filter else "")
                      + (f" | {khoan_filter}" if khoan_filter else "")
                      + (f" | query='{_preview(content_query, 60)}'" if content_query else ""))

                extra_results = search_by_doc_ref(
                    doc_ref=doc_ref,
                    chunk_map=chunk_map,
                    faiss_id_map=faiss_id_map,
                    index=index,
                    article_index_map=article_index_map,
                    dieu_filter=dieu_filter,
                    khoan_filter=khoan_filter,
                    content_query=content_query or question,
                    top_k=10,
                )

                existing_chunk_ids = {r["chunk_id"] for r in ref_index}
                new_results = [r for r in extra_results if r["chunk_id"] not in existing_chunk_ids]

                tool_blocks = []
                for res in new_results:
                    meta     = res.get("metadata", {})
                    chunk_id = res.get("chunk_id", "")
                    doc_num  = (meta.get("doc_num") or "").strip()
                    doc_name = build_doc_name(meta)
                    dieu_so  = (meta.get("dieu_so") or "").strip()
                    article  = (meta.get("article") or dieu_so).strip()
                    year     = parse_year_from_doc_num(doc_num)
                    text     = chunks_text_map.get(chunk_id, build_chunk_text(res, {}))

                    ref_index.append({
                        "ref_id":   next_ref_id,
                        "doc_num":  doc_num,
                        "doc_name": doc_name,
                        "article":  article,
                        "dieu_so":  dieu_so,
                        "year":     year,
                        "text":     text,
                        "chunk_id": chunk_id,
                        "title":    meta.get("title"),
                    })

                    header = f"[{next_ref_id}] {doc_name}"
                    if article:
                        header += f" | {article}"
                    if year:
                        header += f" (năm {year})"
                    if doc_num:
                        header += f" | số hiệu: {doc_num}"
                    tool_blocks.append(f"{header}\n{text}")
                    next_ref_id += 1

                if tool_blocks:
                    tool_result = "\n\n---\n\n".join(tool_blocks)
                    _info(f"  → Tìm thêm được {len(new_results)} chunk(s) mới (ref [{next_ref_id - len(new_results)}..{next_ref_id - 1}])")
                else:
                    tool_result = f"(Không tìm thấy văn bản khớp với '{doc_ref}')"
                    _info(f"  → Không tìm thấy văn bản '{doc_ref}'")

            else:
                # FIX: trước đây chỉ log warning, không trả lỗi cho model nên
                # model dễ lặp lại y nguyên tool call sai ở round sau. Giờ trả
                # lỗi rõ ràng qua tool result để model biết và tự sửa.
                tool_result = (
                    f"LỖI: tool '{fn_name}' không khả dụng ở bước này. Ở bước trả lời này, "
                    f"CHỈ được phép gọi tool 'search_referenced_document' (nếu cần tra cứu "
                    f"thêm văn bản), hoặc trả lời thẳng bằng văn bản thuần nếu không cần "
                    f"tra cứu thêm."
                )
                _warn(f"Tool không xác định: {fn_name}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      tool_result,
            })

        if round_idx == max_tool_rounds:
            _warn("Đã đạt giới hạn tool rounds, buộc LLM trả lời...")
            messages.append({
                "role":    "user",
                "content": "Hãy tổng hợp câu trả lời cuối cùng dựa trên tất cả tài liệu đã tìm được. Không gọi thêm tool. /no_think",
            })

            forced_resp = None
            last_error = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    forced_resp = chat_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        extra_body=NO_THINK_EXTRA_BODY,
                    )
                    break
                except Exception as e:
                    last_error = e
                    _warn(f"Lỗi gọi LLM trả lời (forced, lần {attempt}/{MAX_RETRIES}): {e}")

            if forced_resp is not None:
                final_answer_text = (forced_resp.choices[0].message.content or "").strip()
            else:
                _warn(f"Gọi LLM trả lời (forced) thất bại sau {MAX_RETRIES} lần: {last_error}")
                final_answer_text = ""

            messages.append({"role": "assistant", "content": final_answer_text})
            _done("LLM trả lời (forced)", f"{len(final_answer_text)} ký tự")
            _info(f"Preview: {_preview(final_answer_text, 150)}")
            break

    # ═══════════════════════════════════════════════════════
    # Giai đoạn 3: used_refs — ÉP schema bằng response_format
    # ═══════════════════════════════════════════════════════
    _step(4, "LLM khai báo used_refs (response_format json_schema, cùng conversation)")
    used_refs: list[int] = []

    if ref_index:
        ref_summary = "\n".join(
            f"  [{r['ref_id']}] {r['doc_num']}"
            + (f" | {r['dieu_so']}" if r.get("dieu_so") else "")
            + (f" (năm {r['year']})" if r.get("year") else "")
            for r in ref_index
        )
        _info(f"Tổng refs trong context: {len(ref_index)}")

        messages.append({
            "role": "user",
            "content": (
                f"Dựa trên câu trả lời vừa rồi, hãy xác định những tài liệu nào (theo số [N]) "
                f"đã được sử dụng hoặc trích dẫn.\n\n"
                f"Nếu như có 2 văn bản khác nhau, thì chỉ chọn văn bản có năm mới nhất là được.\n\n"
                f"Danh sách tài liệu:\n{ref_summary} /no_think"
            ),
        })

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                refs_resp = chat_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    response_format=USED_REFS_RESPONSE_FORMAT,
                    extra_body=NO_THINK_EXTRA_BODY,
                )
                raw = (refs_resp.choices[0].message.content or "").strip()
                parsed  = json.loads(raw)
                raw_ids = parsed.get("used_refs", [])

                for x in raw_ids:
                    try:
                        used_refs.append(int(x))
                    except (ValueError, TypeError):
                        pass

                messages.append({"role": "assistant", "content": raw})
                _done("used_refs", str(used_refs[:10]) + ("..." if len(used_refs) > 10 else "")
                      + f" (lần thử {attempt}/{MAX_RETRIES})")
                break

            except Exception as e:
                last_error = e
                used_refs = []
                _warn(f"Lỗi gọi/parse used_refs (lần {attempt}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES:
                    _warn(f"Hết số lần retry cho used_refs: {last_error}. Fallback dùng tất cả.")
                    used_refs = list(range(1, len(ref_index) + 1))
    else:
        _info("Không có ref_index → bỏ qua bước này.")

    return sub_queries, final_answer_text, used_refs, ref_index, context_text


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  RAG Pipeline – Pháp luật Việt Nam")
    print("═" * 60)

    print("\n[INIT] Loading FAISS index & maps...")
    index        = load_index()
    faiss_id_map = load_faiss_id_map()
    chunk_map    = load_chunk_map()
    article_index_map = load_article_index_map()

    chunks_text_map: dict[str, str] = {}
    chunks_json = os.path.join("data", "chunks.json")
    if os.path.exists(chunks_json):
        print("[INIT] Loading embed_text từ chunks.json...")
        with open(chunks_json, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        for c in all_chunks:
            chunks_text_map[c["chunk_id"]] = c.get("embed_text", "")
        print(f"[INIT] ✓ {len(chunks_text_map):,} chunks sẵn sàng.\n")
    else:
        print("[INIT] ⚠ Không tìm thấy chunks.json – dùng metadata rebuild.\n")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INIT] Loaded {len(data)} câu hỏi từ {INPUT_FILE}")

    # ── Resume logic ─────────────────────────────────────────
    results: list[dict] = []
    done_ids: set = set()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                done_ids = {r["id"] for r in results}
                print(f"[RESUME] Tìm thấy {OUTPUT_FILE} – đã có {len(done_ids)}/{len(data)} câu.")
                if done_ids:
                    print(f"[RESUME] IDs đã xong: {sorted(done_ids)}")
                remaining = [item for item in data if item["id"] not in done_ids]
                print(f"[RESUME] Còn lại {len(remaining)} câu cần xử lý.\n")
            except Exception as e:
                print(f"[RESUME] ⚠ Không đọc được {OUTPUT_FILE}: {e} – bắt đầu lại từ đầu.\n")
                results = []
                done_ids = set()
    else:
        print(f"[RESUME] Không tìm thấy {OUTPUT_FILE} – bắt đầu mới.\n")
    # ─────────────────────────────────────────────────────────

    total = len(data)
    processed_count = 0

    for item_idx, item in enumerate(data, start=1):
        q_id     = item["id"]
        question = item["question"]

        # ── Skip nếu đã xử lý ───────────────────────────────
        if q_id in done_ids:
            print(f"  [{item_idx}/{total}] ID={q_id} – ⏭ Đã có, bỏ qua.")
            continue

        print(f"\n{'═' * 60}")
        print(f"  [{item_idx}/{total}] ID={q_id}  (còn lại: {total - len(done_ids) - processed_count - 1})")
        print(f"  Câu hỏi: {_preview(question, 100)}")
        print(f"{'═' * 60}")

        # ── Sub-queries + FAISS search + LLM answer + used_refs ──
        # Toàn bộ 3 giai đoạn LLM (sub-queries, answer/tool-call, used_refs)
        # chạy trong CÙNG MỘT conversation bên trong llm_full_pipeline().
        # search_fn là closure thực hiện Bước 2 (FAISS + Rerank, không phải
        # lệnh gọi LLM) ngay sau khi có sub-queries từ giai đoạn 1.
        def search_fn(sub_queries: list[str]) -> tuple[str, list[dict]]:
            _step(2, f"Embedding search + Rerank ({1 + len(sub_queries)} queries)")
            all_candidates: list[dict] = []
            seen_chunk_ids: set = set()

            for sq_idx, sq in enumerate([question] + sub_queries):
                label = "câu gốc" if sq_idx == 0 else f"sub-query {sq_idx}"
                _info(f"Searching {label}: '{_preview(sq, 80)}'")

                # Port từ v1: initial_k/final_k/rerank_threshold rộng hơn để
                # tăng recall cho retrieval đa-hop (v2 cũ dùng 10/5/0.6 quá hẹp).
                sq_results = search_and_rerank(
                    query=sq,
                    index=index,
                    faiss_id_map=faiss_id_map,
                    chunk_map=chunk_map,
                    initial_k=50,
                    final_k=30,
                    rerank_threshold=0.25,
                )
                before = len(all_candidates)
                for res in sq_results:
                    cid = res.get("chunk_id", "")
                    if cid and cid in seen_chunk_ids:
                        continue
                    if cid:
                        seen_chunk_ids.add(cid)
                    all_candidates.append(res)
                added = len(all_candidates) - before
                _info(f"  → +{added} chunks mới (total: {len(all_candidates)})")

            _done("Search hoàn tất", f"{len(all_candidates)} chunks sau dedup")

            # Tóm tắt top docs để debug
            top_docs = {}
            for c in all_candidates[:10]:
                dn = c.get("metadata", {}).get("doc_num", "?")
                ds = c.get("metadata", {}).get("dieu_so", "")
                top_docs[f"{dn} {ds}".strip()] = True
            _info("Top chunks: " + " | ".join(list(top_docs.keys())[:5]))

            if all_candidates:
                return build_context_with_index(all_candidates, chunks_text_map)
            return "Không tìm thấy tài liệu liên quan đủ độ tin cậy.", []

        sub_queries, answer, used_refs, ref_index, context_text = llm_full_pipeline(
            question=question,
            index=index,
            faiss_id_map=faiss_id_map,
            chunk_map=chunk_map,
            article_index_map=article_index_map,
            chunks_text_map=chunks_text_map,
            search_fn=search_fn,
            max_tool_rounds=3,
        )

        # ── Bước 5: Build output ─────────────────────────────
        _step(5, "Build relevant_docs / relevant_articles")
        relevant_docs, relevant_articles = build_relevant_from_used_refs(used_refs, ref_index)
        _done("relevant_docs", str(len(relevant_docs)))
        for d in relevant_docs[:3]:
            _info(f"  • {_preview(d, 100)}")
        if len(relevant_docs) > 3:
            _info(f"  ... và {len(relevant_docs) - 3} văn bản khác")

        _done("relevant_articles", str(len(relevant_articles)))
        for a in relevant_articles[:3]:
            _info(f"  • {_preview(a, 100)}")
        if len(relevant_articles) > 3:
            _info(f"  ... và {len(relevant_articles) - 3} điều khác")

        results.append({
            "id":                q_id,
            "question":          question,
            "answer":            answer,
            "relevant_docs":     relevant_docs,
            "relevant_articles": relevant_articles,
        })
        done_ids.add(q_id)
        processed_count += 1

        # ── Checkpoint ───────────────────────────────────────
        tmp_file = OUTPUT_FILE + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, OUTPUT_FILE)
        print(f"\n  💾 Checkpoint ({len(done_ids)}/{total}) → {OUTPUT_FILE}")

    print(f"\n{'═' * 60}")
    print(f"  ✅ Hoàn thành! {len(results)} câu hỏi → {OUTPUT_FILE}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()