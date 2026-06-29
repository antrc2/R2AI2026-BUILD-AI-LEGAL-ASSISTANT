"""
main_v2.py – Pipeline RAG pháp luật Việt Nam (icllmlib.LLM, KHÔNG có tool-calling thật)

BẢN REFACTOR – mục tiêu: tăng Articles F2-Macro (ưu tiên giảm over-citation để tăng
precision, vẫn giữ recall) và giảm số lần gọi LLM/câu hỏi.

So với bản cũ, các thay đổi chính:
  1. BỎ bước hỏi LLM riêng "used_refs đã dùng những ref nào" (nguồn gây nhiễu lớn nhất,
     vì model có xu hướng đoán rộng tay, và bản cũ còn fallback "dùng tất cả" khi parse lỗi).
     → Thay bằng trích trực tiếp các số [N] THẬT có trong câu trả lời bằng regex.
     → Tiết kiệm 1 LLM call/câu, và relevant_docs/relevant_articles giờ phản ánh ĐÚNG
       những gì câu trả lời thực sự trích dẫn (cũng khớp với cách BTC nói sẽ tự trích
       điều luật từ chính câu trả lời để chấm).
  2. Retrieval: thay vì rerank riêng từng sub-query (nhiều noise, nhiều lần gọi reranker),
     gộp toàn bộ candidate (FAISS only) từ mọi (sub)query vào 1 pool, rồi RERANK 1 LẦN
     DUY NHẤT theo câu hỏi gốc, áp dụng threshold + final_k tại đây. Vừa giảm nhiễu,
     vừa giảm số lần gọi reranker.
  3. Bước "tra cứu thêm tài liệu" (need_lookup) – vốn đổ thêm rất nhiều chunk không qua
     threshold – được TẮT MẶC ĐỊNH (ENABLE_FOLLOWUP_LOOKUP=False) vì hiện tại recall đã
     khá tốt, precision mới là vấn đề. Có thể bật lại nếu sau này recall tụt.
  4. Sửa build_doc_name(): bản cũ chỉ trả "title" thô, thiếu tiền tố "Loại văn bản +
     Mã văn bản" theo đúng format BTC yêu cầu. Đã thêm suy luận loại văn bản từ mã số.
  5. System prompt câu trả lời được viết lại để: (a) bắt buộc nêu rõ tên văn bản + số
     hiệu + điều NGAY TRONG CÂU (không chỉ dựa vào [N]), (b) nói rõ "đầy đủ" KHÔNG có
     nghĩa là trích dẫn nhiều văn bản, (c) có cấu trúc rõ ràng + một đoạn áp dụng thực
     tế, để phục vụ 4 tiêu chí BTC đọc tay (chính xác, đầy đủ, thực tiễn, rõ ràng).
  6. OUTPUT_FILE đổi thẳng thành "results.json" và thêm package_submission() để tự
     đóng gói thành results.zip (zip phẳng) đúng định dạng nộp bài.

LƯU Ý CẦN BẠN TỰ KIỂM TRA:
  - DOC_TYPE_PATTERNS trong build_doc_name() là suy luận heuristic từ mã văn bản
    (ví dụ .../QH.. -> "Luật"). Đây là best-effort, hãy lấy vài chunk thật để kiểm tra
    field `title`/`doc_num` trong metadata của bạn có khớp giả định không, và chỉnh
    lại mapping nếu cần.
  - Các hằng số ở khối "Retrieval tuning knobs" là chỗ chính để bạn chỉnh sau mỗi lần
    nộp bài, xem phần comment cạnh từng hằng số.
"""

import os
import json
import re
import time
import zipfile
from typing import Any

import faiss

from icllmlib import LLM

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
MODEL_LLM   = "llm3.1-sea"   # tên alias nội bộ – model thật chạy phía sau là Qwen3

INPUT_FILE  = "R2AIStage1DATA.json"
OUTPUT_FILE = "results.json"     # ĐÚNG tên file nộp bài theo yêu cầu BTC
SUBMISSION_ZIP = "results.zip"   # zip phẳng chứa results.json, tự tạo cuối main()

RETRY_ATTEMPTS = 5
RETRY_DELAY    = 0.5  # giây

# ── Retrieval tuning knobs ───────────────────────────────────
# Đây là những chỗ NÊN chỉnh đầu tiên dựa trên kết quả leaderboard kỳ sau.
PER_QUERY_FAISS_K  = 45    # số candidate thô (chỉ FAISS, chưa rerank) lấy về MỖI (sub)query.
                            # Tăng nếu RECALL thấp, không ảnh hưởng nhiều đến precision
                            # (vì còn bị lọc lại ở bước rerank+threshold phía sau).
GLOBAL_RERANK_CAP  = 200   # giới hạn pool trước khi đưa vào reranker (tránh quá tải).
FINAL_CONTEXT_K    = 20    # số chunk TỐI ĐA được đưa vào context cho LLM trả lời, sau rerank.
                            # Giảm số này → ít chunk nhiễu trong context → tăng precision,
                            # nhưng có thể giảm recall nếu câu hỏi cần nhiều điều luật khác nhau.
RERANK_THRESHOLD   = 0.35  # Lọc theo rerank_score sau khi rerank 1 lần theo câu hỏi gốc.
                            # TĂNG ngưỡng này → precision cao hơn, recall thấp hơn (và ngược lại).
                            # Đây là nút chỉnh ảnh hưởng trực tiếp nhất tới Articles Precision.

# ── Followup lookup (tra cứu thêm văn bản được nhắc tới trong câu trả lời sơ bộ) ──
# Bước này giúp RECALL (đuổi theo các văn bản dẫn chiếu) nhưng dễ làm giảm PRECISION
# vì các chunk thêm vào không qua threshold rerank. Hiện tại recall đã khá ổn
# (0.707) còn precision đang là nút cổ chai (0.317) → tắt mặc định.
ENABLE_FOLLOWUP_LOOKUP = True
MAX_FOLLOWUP_ROUNDS    = 2   # chỉ dùng nếu ENABLE_FOLLOWUP_LOOKUP=True
FOLLOWUP_TOPK          = 6   # giảm từ 20 (bản cũ) xuống 6 để hạn chế noise nếu bật lại

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
# Lỗi
# ─────────────────────────────────────────────────────────────

class LLMClientError(RuntimeError):
    """Lỗi gốc cho mọi vấn đề liên quan tới gọi/parse kết quả LLM."""
    pass


class LLMCallError(LLMClientError):
    """Lỗi khi gọi LLM (network, response rỗng, hết retry, v.v.)."""
    pass


def _call_llm(prompt: str, system_prompt: str, max_decoding_length: int = 2048,
              temperature: float = 0.1,
              retries: int = RETRY_ATTEMPTS, delay: int = RETRY_DELAY) -> str:
    """
    Gọi icllmlib.LLM.generate() và trả về answer_norm (string).
    Retry tối đa `retries` lần nếu request lỗi hoặc trả về rỗng.
    """
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
                _warn(f"Gọi LLM lỗi (lần {attempt}/{retries}): {e}. Thử lại sau {delay}s...")
                time.sleep(delay)
            else:
                _warn(f"Gọi LLM lỗi sau {retries} lần thử: {e}")

    raise LLMCallError(f"Gọi LLM thất bại sau {retries} lần: {last_error}")


JSON_FORMAT_RULES = """
QUY TẮC BẮT BUỘC VỀ FORMAT JSON:
- Chỉ trả về DUY NHẤT một JSON object hợp lệ, không kèm bất kỳ văn bản nào khác.
- Không dùng markdown code block (không có ```json hoặc ```).
- Không thêm giải thích, không thêm ghi chú trước hoặc sau JSON.
- Toàn bộ JSON phải nằm trên một khối liền mạch, mở bằng { và đóng bằng }.
- Dùng dấu ngoặc kép " cho key và string value, không dùng dấu nháy đơn.
- Không để dấu phẩy dư (trailing comma) trước } hoặc ].
- Đảm bảo mọi chuỗi (string) được đóng mở dấu ngoặc kép đầy đủ, không bị ngắt giữa dòng.
- Nếu một giá trị có chứa dấu ngoặc kép, phải escape bằng \\" để không phá vỡ cấu trúc JSON.
- Output phải parse được trực tiếp bằng json.loads() của Python.
"""


def _parse_json_answer(answer: str) -> dict[str, Any]:
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
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON response must be an object.")
    return parsed


def _call_llm_json(prompt: str, system_prompt: str, max_decoding_length: int = 1024,
                    temperature: float = 0.1,
                    retries: int = RETRY_ATTEMPTS, delay: int = RETRY_DELAY) -> dict:
    """
    Gọi LLM và parse JSON bằng _parse_json_answer(). Nếu request lỗi HOẶC
    parse JSON lỗi, retry lại toàn bộ (gọi LLM lại) tối đa `retries` lần.
    """
    full_system_prompt = f"{system_prompt}\n{JSON_FORMAT_RULES}"
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            raw = _call_llm(
                prompt, full_system_prompt,
                max_decoding_length=max_decoding_length,
                temperature=temperature,
                retries=1,  # retry ở tầng JSON sẽ tự gọi lại _call_llm, không cần lồng retry
            )
            return _parse_json_answer(raw)

        except Exception as e:
            last_error = e
            if attempt < retries:
                _warn(f"Gọi/Parse JSON lỗi (lần {attempt}/{retries}): {e}. Thử lại sau {delay}s...")
                time.sleep(delay)
            else:
                _warn(f"Gọi/Parse JSON lỗi sau {retries} lần thử: {e}")

    raise LLMClientError(f"Gọi/Parse JSON thất bại sau {retries} lần: {last_error}")


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


# Suy luận "Loại văn bản" từ mã số, theo thứ tự ưu tiên (pattern cụ thể trước).
# ★ Đây là heuristic – hãy kiểm tra với dữ liệu thật và chỉnh lại nếu cần.
DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
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
    """
    Trả về tên văn bản đúng công thức BTC yêu cầu:
        Loại văn bản + Mã văn bản + Trích yếu
    Ví dụ: doc_num="04/2017/QH14", title="Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
           -> "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
    """
    title = (meta.get("title") or "").strip()
    doc_num = (meta.get("doc_num") or "").strip()

    if not title:
        return doc_num

    if doc_num and doc_num in title:
        # title đã tự chứa mã văn bản -> coi như đã đủ thông tin, giữ nguyên
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
            "title": meta.get('title')
        })

        header = f"[{i}] {doc_name}"
        if article:
            header += f" | {article}"
        if year:
            header += f" (năm {year})"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks), ref_index


def _rebuild_context_blocks(ref_index: list[dict]) -> str:
    """Dựng lại context_text từ ref_index hiện tại (dùng khi bổ sung ref mới ở followup)."""
    blocks = []
    for r in ref_index:
        header = f"[{r['ref_id']}] {r['doc_name']}"
        if r.get("article"):
            header += f" | {r['article']}"
        if r.get("year"):
            header += f" (năm {r['year']})"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


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
# Trích used_refs trực tiếp từ câu trả lời (KHÔNG gọi LLM thêm)
# ─────────────────────────────────────────────────────────────

# Chỉ chấp nhận [N] hoặc [N, M, ...] – KHÔNG chấp nhận khoảng [1-3] (model được dặn
# không dùng định dạng khoảng, xem ANSWER_SYSTEM_PROMPT).
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_used_refs_from_answer(answer: str, max_ref_id: int) -> list[int]:
    """
    Trích các số ref [N] THẬT xuất hiện trong câu trả lời, theo đúng thứ tự xuất hiện,
    loại bỏ trùng và loại bỏ số ngoài phạm vi ref_index hiện có (chống LLM bịa số).
    """
    used: list[int] = []
    seen: set[int] = set()
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


# ─────────────────────────────────────────────────────────────
# LLM calls (icllmlib, không có function calling thật)
# ─────────────────────────────────────────────────────────────

def llm_sub_queries(question: str) -> list[str]:
    system_prompt = "Trả về valid JSON object, không markdown, không giải thích thêm."

    prompt = f"""Bạn là chuyên gia phân tích cú pháp câu hỏi pháp luật Việt Nam.

Câu hỏi gốc: "{question}"

Nhiệm vụ: PHÂN RÃ (decompose) câu hỏi gốc thành các câu hỏi con, KHÔNG phải tạo thêm ý mới.

Quy tắc bắt buộc:
- Mỗi sub-query CHỈ ĐƯỢC chứa thông tin/thực thể/điều kiện đã CÓ SẴN trong câu hỏi gốc.
- TUYỆT ĐỐI KHÔNG suy diễn ra chủ thể, hậu quả, hay khía cạnh pháp lý mà câu hỏi gốc không
  đề cập (ví dụ: câu gốc hỏi về "công ty nộp chậm tiền thuế" thì KHÔNG được tự suy ra câu hỏi
  về "trách nhiệm người đại diện pháp luật" hay "lãi suất chậm nộp" nếu câu gốc không hỏi điều đó).
- Nếu câu hỏi gốc CHỈ chứa một ý duy nhất (một hành vi, một điều kiện, một đối tượng) và không
  thể tách nhỏ hơn mà không làm mất hoặc bóp méo nghĩa, hãy trả về CHÍNH XÁC 1 sub-query là
  câu hỏi gốc được viết lại rõ ràng hơn (chuẩn hóa thuật ngữ pháp lý), không thêm nội dung mới.
- Nếu câu hỏi gốc ghép nhiều ý (ví dụ: vừa hỏi về hành vi vi phạm vừa hỏi về mức xử phạt vừa hỏi
  về thời hạn), hãy tách mỗi ý thành 1 sub-query riêng, mỗi câu là một CÂU HOÀN CHỈNH bằng tiếng Việt.
- Giữ nguyên core subject (chủ thể + hành vi chính) của câu gốc trong MỌI sub-query.
- Không thêm ví dụ minh họa, không thêm trường hợp giả định, không mở rộng phạm vi câu hỏi.
- Số lượng sub-query phụ thuộc vào số ý có thật trong câu hỏi gốc, KHÔNG cố tạo ra càng nhiều
  câu càng tốt. Thường chỉ 1-3 câu là đủ; chỉ tạo nhiều hơn nếu câu gốc thực sự ghép nhiều ý.

Ví dụ ĐÚNG:
Câu gốc: "Doanh nghiệp nhỏ và vừa được ưu đãi thuế thu nhập doanh nghiệp như thế nào, và thủ tục đăng ký hưởng ưu đãi ra sao?"
→ {{"sub_queries": [
    "Doanh nghiệp nhỏ và vừa được ưu đãi về thuế thu nhập doanh nghiệp như thế nào",
    "Thủ tục đăng ký hưởng ưu đãi thuế thu nhập doanh nghiệp đối với doanh nghiệp nhỏ và vừa"
  ]}}

Ví dụ SAI (KHÔNG làm theo - đây là việc "tư vấn mở rộng" chứ không phải phân rã):
Câu gốc: "Công ty tự tính thuế nhưng nộp chậm hơn ngày cuối cùng của thời hạn nộp hồ sơ khai thuế thì bị xử phạt thế nào?"
→ Sai vì tự suy ra các câu KHÔNG có trong câu gốc như:
  - "Trách nhiệm pháp lý của người đại diện pháp luật khi doanh nghiệp nộp thuế chậm hạn" (KHÔNG được hỏi)
  - "Lãi suất chậm nộp tiền thuế áp dụng khi doanh nghiệp nộp thuế sau hạn" (KHÔNG được hỏi, đây là suy diễn)
  - "Đối tượng chịu xử phạt trong trường hợp..." (KHÔNG được hỏi, đây là góc nhìn khác do LLM tự thêm)
Đúng ra với câu hỏi này chỉ nên trả về 1 sub-query (vì chỉ có 1 ý: nộp chậm thuế tự tính thì bị xử phạt gì):
→ {{"sub_queries": ["Công ty tự tính thuế nộp chậm so với thời hạn nộp hồ sơ khai thuế thì bị xử phạt vi phạm hành chính như thế nào"]}}

Trả về JSON object, không giải thích:
{{"sub_queries": ["câu hỏi con 1", "câu hỏi con 2"]}}"""

    try:
        data = _call_llm_json(prompt, system_prompt, max_decoding_length=1024)
        sqs = data.get("sub_queries", [])
        return sqs if sqs else [question]
    except Exception as e:
        _warn(f"Lỗi sinh sub-queries sau retry: {e}. Fallback dùng câu hỏi gốc.")
        return [question]


def llm_search_referenced_document(
    question: str,
    answer_so_far: str,
    ref_index: list[dict],
    index: faiss.Index,
    faiss_id_map: dict,
    chunk_map: dict,
    chunks_text_map: dict,
    article_index_map: dict,
) -> tuple[list[dict], bool]:
    """
    [CHỈ DÙNG KHI ENABLE_FOLLOWUP_LOOKUP=True]
    Hỏi LLM xem có cần tra cứu thêm văn bản nào không (thay cho function calling).
    Trả về (new_ref_entries, has_more_lookup).
    """
    ref_summary = "\n".join(
        f"  [{r['ref_id']}] {r['doc_num']} {r.get('article', '')}".strip()
        for r in ref_index
    ) or "(không có)"

    system_prompt = (
        "Bạn là trợ lý xác định nhu cầu tra cứu thêm tài liệu pháp luật. "
        "Chỉ trả về JSON object hợp lệ, không markdown, không giải thích."
    )

    prompt = f"""CÂU HỎI: {question}

CÂU TRẢ LỜI ĐANG SOẠN (dựa trên tài liệu hiện có):
{answer_so_far}

DANH SÁCH TÀI LIỆU ĐÃ CÓ:
{ref_summary}

Nếu câu trả lời trên có trích dẫn/đề cập đến một văn bản pháp luật KHÁC mà chưa có trong danh sách trên,
hãy liệt kê văn bản đó để tra cứu thêm. Nếu không cần tra cứu thêm, trả về danh sách rỗng.
CHỈ liệt kê khi thực sự cần thiết để trả lời đúng câu hỏi – không tra cứu thêm "cho chắc".

Lưu ý:
- "dieu_filter": chỉ ghi số điều, ví dụ "Điều 74". Để rỗng nếu không xác định được điều cụ thể.
- "khoan_filter": chỉ ghi số khoản, ví dụ "Khoản 3". Để rỗng nếu không có hoặc cần cả điều.
- KHÔNG gộp điều và khoản vào cùng một field.

Trả về đúng JSON object, không kèm gì khác:
{{"need_lookup": [{{"doc_ref": "số hiệu văn bản (Ví dụ: 04/2017/QH14)", "dieu_filter": "Điều X hoặc rỗng", "khoan_filter": "Khoản Y hoặc rỗng", "content_query": "từ khóa cần tìm hoặc rỗng"}}]}}"""

    try:
        parsed = _call_llm_json(prompt, system_prompt, max_decoding_length=1024)
        need_lookup = parsed.get("need_lookup", [])
    except Exception as e:
        _warn(f"Lỗi xác định need_lookup sau retry: {e}. Bỏ qua bước tra cứu thêm.")
        need_lookup = []

    if not need_lookup:
        return [], False

    next_ref_id = len(ref_index) + 1
    new_entries = []
    existing_chunk_ids = {r["chunk_id"] for r in ref_index}

    for lk in need_lookup:
        doc_ref       = lk.get("doc_ref", "")
        dieu_filter   = lk.get("dieu_filter") or None
        khoan_filter  = lk.get("khoan_filter") or None
        content_query = lk.get("content_query") or question
        if not doc_ref:
            continue

        _tool(
            f"search_referenced_document: '{doc_ref}'"
            + (f" | {dieu_filter}" if dieu_filter else "")
            + (f" | {khoan_filter}" if khoan_filter else "")
            + f" | query='{_preview(content_query, 60)}'"
        )

        extra_results = search_by_doc_ref(
            doc_ref=doc_ref,
            chunk_map=chunk_map,
            faiss_id_map=faiss_id_map,
            index=index,
            article_index_map=article_index_map,
            dieu_filter=dieu_filter,
            khoan_filter=khoan_filter,
            content_query=content_query,
            top_k=FOLLOWUP_TOPK,
        )

        for res in extra_results:
            chunk_id = res.get("chunk_id", "")
            if chunk_id in existing_chunk_ids:
                continue
            existing_chunk_ids.add(chunk_id)

            meta     = res.get("metadata", {})
            doc_num  = (meta.get("doc_num") or "").strip()
            doc_name = build_doc_name(meta)

            dieu_so  = (meta.get("dieu_so") or "").strip()
            article  = (meta.get("article") or dieu_so).strip()
            if not dieu_so:
                dieu_so = article

            year     = parse_year_from_doc_num(doc_num)
            text     = chunks_text_map.get(chunk_id, build_chunk_text(res, {}))

            new_entries.append({
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
            next_ref_id += 1

    if new_entries:
        _info(f"  → Tìm thêm được {len(new_entries)} chunk(s) mới.")
    else:
        _info("  → Không tìm thấy chunk mới nào khớp.")

    return new_entries, True


ANSWER_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp luật Việt Nam, trả lời cho người dùng phổ thông.

NGUYÊN TẮC NỘI DUNG (bắt buộc):
1. CHỈ sử dụng thông tin có trong TÀI LIỆU THAM KHẢO được cung cấp. Không suy đoán, không
   dùng kiến thức ngoài tài liệu, không bịa số liệu/mức phạt/thời hạn không có trong tài liệu.
2. TRƯỚC KHI trích dẫn một điều khoản, kiểm tra: đối tượng/phạm vi áp dụng của điều khoản đó
   có khớp với đối tượng/tình huống trong câu hỏi không. Nếu điều khoản nói về một đối tượng
   khác (ví dụ: doanh nghiệp trong khu công nghiệp, trong khi câu hỏi hỏi về doanh nghiệp nhỏ
   và vừa nói chung), KHÔNG trích dẫn điều khoản đó, dù chủ đề chung có vẻ liên quan.
3. "Đầy đủ và toàn diện" nghĩa là trả lời HẾT các khía cạnh THỰC SỰ được hỏi trong câu hỏi gốc,
   KHÔNG có nghĩa là trích dẫn càng nhiều văn bản/điều khoản càng tốt. Một câu trả lời trích
   đúng 2-3 điều khoản còn tốt hơn nhiều một câu trả lời trích 10 điều trong đó có nhiều điều
   không thực sự áp dụng. Nếu không chắc một điều khoản có áp dụng hay không, HÃY BỎ QUA.
4. Nếu có 2 văn bản/điều khoản nội dung trùng nhau (văn bản cũ đã bị thay thế), chỉ dùng văn
   bản/điều khoản có năm ban hành mới nhất.

NGUYÊN TẮC TRÍCH DẪN (bắt buộc):
- Mỗi khi nêu một quy định, PHẢI nêu rõ TRONG CÙNG CÂU: loại văn bản + số hiệu văn bản + số
  điều, rồi đặt số thứ tự tài liệu trong dấu vuông [N] ngay sau đó.
  Ví dụ ĐÚNG: "Theo Điều 5 Luật Hỗ trợ doanh nghiệp nhỏ và vừa (Luật số 04/2017/QH14) [1], ..."
  Ví dụ SAI (thiếu tên/số hiệu văn bản): "Theo quy định [1], doanh nghiệp được..."
- Định dạng số tài liệu: [N] cho một tài liệu, hoặc [N, M] khi một câu dựa trên nhiều tài liệu
  cùng lúc. KHÔNG dùng định dạng khoảng (ví dụ: [1-3]).
- KHÔNG bịa ra số thứ tự [N] không có trong TÀI LIỆU THAM KHẢO.
- KHÔNG đặt [N] cho một câu nếu nội dung câu đó không thực sự dựa vào tài liệu [N] đó.

NGUYÊN TẮC HÌNH THỨC (để dễ đọc & áp dụng thực tế):
- Câu đầu tiên: trả lời TRỰC TIẾP và ngắn gọn vào trọng tâm câu hỏi.
- Các đoạn sau: trình bày từng điều kiện/quy định liên quan, mỗi đoạn ngắn (2-4 câu), có trích
  dẫn đầy đủ như trên.
- Nếu phù hợp, thêm một đoạn ngắn cuối giải thích cách áp dụng thực tế (ví dụ: hồ sơ cần gì, ai
  chịu trách nhiệm, mức phạt/thời hạn cụ thể là bao nhiêu...) dựa đúng trên tài liệu đã trích.
- Dùng tiếng Việt rõ ràng; nếu phải dùng thuật ngữ pháp lý, giải thích ngắn gọn nghĩa của nó.
- Trả lời bằng văn bản thuần (không markdown code block, không JSON, không dùng ký hiệu *, #).
  Có thể dùng dấu "-" để liệt kê nếu câu hỏi có nhiều mục."""


def llm_answer_and_refs(
    question: str,
    context_text: str,
    index: faiss.Index,
    faiss_id_map: dict,
    chunk_map: dict,
    chunks_text_map: dict,
    article_index_map: dict,
    ref_index: list[dict],
) -> tuple[str, list[int], list[dict]]:
    # ── Giai đoạn 1: Sinh câu trả lời ────────────────────────
    _step(3, "LLM sinh câu trả lời")

    base_prompt = f"""TÀI LIỆU THAM KHẢO (đánh số để trích dẫn):
{context_text}

---

CÂU HỎI: {question}

Hãy trả lời dựa trên tài liệu, tuân thủ đúng các nguyên tắc trích dẫn ở trên."""

    try:
        answer = _call_llm(base_prompt, ANSWER_SYSTEM_PROMPT, max_decoding_length=3072)
    except Exception as e:
        _warn(f"Lỗi sinh câu trả lời sau retry: {e}")
        answer = ""

    _done("LLM trả lời", f"{len(answer)} ký tự")
    _info(f"Preview: {_preview(answer, 150)}")

    # ── Giai đoạn 1b: Tra cứu thêm (TẮT mặc định, xem ENABLE_FOLLOWUP_LOOKUP) ──
    if ENABLE_FOLLOWUP_LOOKUP:
        for round_idx in range(MAX_FOLLOWUP_ROUNDS):
            new_entries, has_more = llm_search_referenced_document(
                question=question,
                answer_so_far=answer,
                ref_index=ref_index,
                index=index,
                article_index_map=article_index_map,
                faiss_id_map=faiss_id_map,
                chunk_map=chunk_map,
                chunks_text_map=chunks_text_map,
            )
            if not has_more or not new_entries:
                break

            ref_index.extend(new_entries)
            context_text = _rebuild_context_blocks(ref_index)

            retry_prompt = f"""TÀI LIỆU THAM KHẢO (đánh số để trích dẫn, đã bổ sung thêm):
{context_text}

---

CÂU HỎI: {question}

Hãy tổng hợp câu trả lời cuối cùng dựa trên tất cả tài liệu trên, tuân thủ đúng các nguyên tắc trích dẫn ở trên."""

            try:
                answer = _call_llm(retry_prompt, ANSWER_SYSTEM_PROMPT, max_decoding_length=4096)
            except Exception as e:
                _warn(f"Lỗi trả lời lại sau bổ sung tài liệu (round {round_idx + 1}): {e}")
                break

            _info(f"Round {round_idx + 1}: trả lời lại sau khi bổ sung tài liệu, "
                  f"preview: {_preview(answer, 150)}")

    # ── Giai đoạn 2: used_refs – trích trực tiếp từ câu trả lời (KHÔNG gọi LLM) ──
    _step(4, "Trích used_refs từ câu trả lời (regex, không tốn LLM call)")
    used_refs = extract_used_refs_from_answer(answer, max_ref_id=len(ref_index))

    if not used_refs and ref_index and answer:
        _warn("Không thấy [N] nào trong câu trả lời. Thử lại 1 lần với nhắc nhở mạnh hơn.")
        reminder_prompt = base_prompt + (
            "\n\nLƯU Ý QUAN TRỌNG: câu trả lời TRƯỚC ĐÓ của bạn thiếu hoàn toàn ký hiệu trích "
            "dẫn [N]. Lần này BẮT BUỘC phải đánh số [N] ngay sau mỗi nội dung được trích từ "
            "tài liệu tham khảo, theo đúng nguyên tắc trích dẫn đã nêu."
        )
        try:
            answer_retry = _call_llm(reminder_prompt, ANSWER_SYSTEM_PROMPT, max_decoding_length=3072)
            retry_refs = extract_used_refs_from_answer(answer_retry, max_ref_id=len(ref_index))
            if retry_refs:
                answer, used_refs = answer_retry, retry_refs
                _info(f"Retry thành công, tìm thấy {len(retry_refs)} ref.")
        except Exception as e:
            _warn(f"Lỗi retry nhắc trích dẫn: {e}")

    _done("used_refs", str(used_refs[:15]) + ("..." if len(used_refs) > 15 else ""))

    return answer, used_refs, ref_index


# ─────────────────────────────────────────────────────────────
# Đóng gói nộp bài
# ─────────────────────────────────────────────────────────────

def package_submission(json_path: str = OUTPUT_FILE, zip_path: str = SUBMISSION_ZIP) -> None:
    """Nén json_path thành zip PHẲNG (không thư mục con) chứa đúng file 'results.json'."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")
    print(f"\n  📦 Đã đóng gói nộp bài → {zip_path} (chứa results.json)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  RAG Pipeline – Pháp luật Việt Nam")
    print("═" * 60)
    print(f"\n[CONFIG] PER_QUERY_FAISS_K={PER_QUERY_FAISS_K} | FINAL_CONTEXT_K={FINAL_CONTEXT_K} "
          f"| RERANK_THRESHOLD={RERANK_THRESHOLD} | ENABLE_FOLLOWUP_LOOKUP={ENABLE_FOLLOWUP_LOOKUP}")

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

        if q_id in done_ids:
            print(f"  [{item_idx}/{total}] ID={q_id} – ⏭ Đã có, bỏ qua.")
            continue

        print(f"\n{'═' * 60}")
        print(f"  [{item_idx}/{total}] ID={q_id}  (còn lại: {total - len(done_ids) - processed_count - 1})")
        print(f"  Câu hỏi: {_preview(question, 100)}")
        print(f"{'═' * 60}")

        # ── Bước 1: Sub-queries ──────────────────────────────
        _step(1, "Sinh sub-queries (multi-hop)")
        sub_queries = llm_sub_queries(question)
        _done("Sub-queries", f"{len(sub_queries)} queries")
        for i, sq in enumerate(sub_queries, 1):
            _info(f"  [{i}] {_preview(sq, 100)}")

        # ── Bước 2: FAISS (mỗi query) → gộp pool → Rerank 1 LẦN theo câu hỏi gốc ──
        _step(2, f"Retrieval: FAISS x{1 + len(sub_queries)} query → gộp pool → rerank 1 lần")
        pool: list[dict] = []
        seen_chunk_ids: set = set()

        for sq_idx, sq in enumerate([question] + sub_queries):
            label = "câu gốc" if sq_idx == 0 else f"sub-query {sq_idx}"
            _info(f"FAISS search {label}: '{_preview(sq, 80)}'")

            raw_results = faiss_search(
                query=sq,
                index=index,
                faiss_id_map=faiss_id_map,
                chunk_map=chunk_map,
                top_k=PER_QUERY_FAISS_K,
            )
            before = len(pool)
            for res in raw_results:
                cid = res.get("chunk_id", "")
                if cid and cid in seen_chunk_ids:
                    continue
                if cid:
                    seen_chunk_ids.add(cid)
                pool.append(res)
            _info(f"  → +{len(pool) - before} chunk mới (pool: {len(pool)})")

        if len(pool) > GLOBAL_RERANK_CAP:
            _warn(f"Pool {len(pool)} > cap {GLOBAL_RERANK_CAP}, cắt theo điểm FAISS trước khi rerank.")
            pool.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            pool = pool[:GLOBAL_RERANK_CAP]

        _info(f"Rerank toàn bộ pool ({len(pool)} chunk) theo câu hỏi gốc...")
        reranked = rerank(question, pool, top_k=FINAL_CONTEXT_K)
        all_candidates = [c for c in reranked if c.get("rerank_score", 0.0) >= RERANK_THRESHOLD]
        _done("Retrieval hoàn tất", f"{len(all_candidates)}/{len(reranked)} chunk qua threshold {RERANK_THRESHOLD}")

        top_docs = {}
        for c in all_candidates[:10]:
            dn = c.get("metadata", {}).get("doc_num", "?")
            ds = c.get("metadata", {}).get("dieu_so", "")
            top_docs[f"{dn} {ds}".strip()] = True
        _info("Top chunks: " + " | ".join(list(top_docs.keys())[:5]))

        if all_candidates:
            context_text, ref_index = build_context_with_index(all_candidates, chunks_text_map)
        else:
            context_text = "Không tìm thấy tài liệu liên quan đủ độ tin cậy."
            ref_index    = []

        # ── Bước 3 + 4: trả lời + trích used_refs từ chính câu trả lời ───────
        answer, used_refs, ref_index = llm_answer_and_refs(
            question=question,
            context_text=context_text,
            index=index,
            faiss_id_map=faiss_id_map,
            chunk_map=chunk_map,
            article_index_map=article_index_map,
            chunks_text_map=chunks_text_map,
            ref_index=ref_index,
        )

        relevant_docs, relevant_articles = build_relevant_from_used_refs(used_refs, ref_index)

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

    # package_submission(OUTPUT_FILE, SUBMISSION_ZIP)


if __name__ == "__main__":
    main()