"""
search_v2.py – Search module cho pipeline RAG pháp luật Việt Nam (schema mới)

Schema mới (từ chunking):
  data/chunks.json        – list of {chunk_id, embed_text, metadata}
  data/faiss_id_map.json  – {faiss_idx(str) -> chunk_id(str)}
  data/chunk_map.json     – {chunk_id -> metadata}
  data/faiss.index        – FAISS IndexIDMap

Metadata mỗi chunk:
  title, doc_num, doc_id, dieu_so, leaf_level, leaf_title,
  + các field level động: article, chapter, section, ...
"""

import os
import json
import re
import faiss
import numpy as np
import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

EMBEDDING_URL   = "http://127.0.0.1:11113/v1"
EMBEDDING_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"
EMBEDDING_DIM = 1024

RERANKER_URL    = "http://127.0.0.1:11112/v2/rerank"   # FastAPI reranker endpoint
RERANKER_MODEL  = "Qwen/Qwen3-Reranker-0.6B"

DATA_DIR           = "data"
FAISS_INDEX_FILE   = os.path.join(DATA_DIR, "faiss.index")
FAISS_ID_MAP_FILE  = os.path.join(DATA_DIR, "faiss_id_map.json")
CHUNK_MAP_FILE     = os.path.join(DATA_DIR, "chunk_map.json")
ARTICLE_MAP_FILE = os.path.join(DATA_DIR,"article_index_map.json")

embed_client = OpenAI(api_key="dummy", base_url=EMBEDDING_URL)


# ─────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────

def load_index():
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"Không tìm thấy file index: {FAISS_INDEX_FILE}")
    idx = faiss.read_index(FAISS_INDEX_FILE)
    print(f"[FAISS] Loaded: {idx.ntotal:,} vectors")
    return idx

def load_faiss_id_map():
    if not os.path.exists(FAISS_ID_MAP_FILE):
        raise FileNotFoundError(f"Không tìm thấy file map: {FAISS_ID_MAP_FILE}")
    with open(FAISS_ID_MAP_FILE, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def load_chunk_map():
    if not os.path.exists(CHUNK_MAP_FILE):
        raise FileNotFoundError(f"Không tìm thấy file map: {CHUNK_MAP_FILE}")
    with open(CHUNK_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
def load_article_index_map():
    if not os.path.exists(ARTICLE_MAP_FILE):
        raise FileNotFoundError(f"Không tìm thấy file map: {ARTICLE_MAP_FILE}")
    with open(ARTICLE_MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms

def embed_texts(texts: list[str]) -> np.ndarray:
    """Gọi API embedding và chuẩn hóa vector."""
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)
        
    resp = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    data = sorted(resp.data, key=lambda x: x.index)
    vecs = np.array([x.embedding for x in data], dtype=np.float32)
    return _normalize(vecs)


# ─────────────────────────────────────────────────────────────
# FAISS search
# ─────────────────────────────────────────────────────────────

def faiss_search(
    query: str,
    index: faiss.Index,
    faiss_id_map: dict,
    chunk_map: dict,
    top_k: int = 20,
) -> list[dict]:
    """Tìm kiếm semantic toàn cục."""
    vec = embed_texts([query])
    if vec.size == 0:
        return []
        
    scores, ids = index.search(vec, top_k)

    results = []
    for score, faiss_idx in zip(scores[0], ids[0]):
        if faiss_idx < 0:
            continue
        chunk_id = faiss_id_map.get(int(faiss_idx))
        if chunk_id is None:
            continue
        meta = chunk_map.get(chunk_id, {})
        results.append({
            "chunk_id": chunk_id,
            "score":    float(score),
            "metadata": meta,
        })
    return results


# ─────────────────────────────────────────────────────────────
# Reranker
# ─────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """
    Gọi FastAPI reranker (vLLM). 
    Xử lý cấu trúc response: { "results": [ { "index": 0, "relevance_score": float } ] }
    """
    if not candidates:
        return []

    texts = [_build_embed_text(c) for c in candidates]

    try:
        resp = requests.post(
            RERANKER_URL,
            json={
                "query": query, 
                "documents": texts, 
                "model": RERANKER_MODEL
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Lấy danh sách results từ response
        results_list = data.get("results", [])
        
        if not results_list:
            print("[reranker warning] Không có kết quả từ reranker")
            return candidates[:top_k]

        # Sắp xếp lại theo index để đảm bảo thứ tự khớp với mảng 'texts' đầu vào
        sorted_results = sorted(results_list, key=lambda x: x.get("index", 0))
        
        # Trích xuất relevance_score
        scores = [r.get("relevance_score", 0.0) for r in sorted_results]

    except Exception as e:
        print(f"[reranker error] {e} – giữ nguyên thứ tự FAISS")
        return candidates[:top_k]

    # Ghép score vào candidate tương ứng
    paired = list(zip(scores, candidates))
    
    # Sắp xếp giảm dần theo relevance_score
    ranked = sorted(paired, key=lambda x: x[0], reverse=True)
    
    # Gán lại rerank_score vào metadata của candidate để tiện theo dõi
    final_results = []
    for score, cand in ranked[:top_k]:
        cand["rerank_score"] = float(score)
        final_results.append(cand)
        
    return final_results


def _build_embed_text(candidate: dict) -> str:
    """Xây text đại diện từ metadata để rerank."""
    meta = candidate.get("metadata", {})
    parts = []
    if meta.get("title"):
        parts.append(meta["title"])
    # Điều
    art = meta.get("article", "")
    if art:
        parts.append(art)
    # hierarchy thêm
    for lvl in ("chapter", "section", "subsection"):
        v = meta.get(lvl, "")
        if v:
            parts.append(v)
    # leaf_title
    lt = meta.get("leaf_title", "")
    if lt and lt not in parts:
        parts.append(lt)
    return " - ".join(parts)


# ─────────────────────────────────────────────────────────────
# Search theo tên/số hiệu văn bản (title-based)
# ─────────────────────────────────────────────────────────────

def _normalize_doc_ref(text: str) -> str:
    """Chuẩn hoá chuỗi số hiệu / tên văn bản để so sánh mờ."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def _extract_doc_num(text: str) -> str | None:
    """
    Trích số hiệu văn bản thuần từ chuỗi bất kỳ.
    Ví dụ: 'Nghị định 58/2020/NĐ-CP' -> '58/2020/NĐ-CP'
           '36/2015/QĐ-TTg'          -> '36/2015/QĐ-TTg'
    Pattern chung: <số>/<năm>/<mã loại văn bản viết hoa, có thể có dấu gạch ngang>
    """
    if not text:
        return None
    match = re.search(
        r'\d+[A-Za-z]*\s*/\s*\d{4}\s*/\s*[A-ZĐƯƠ]+(?:-[A-ZĐƯƠ]+)*',
        text,
        flags=re.UNICODE,
    )
    if not match:
        return None
    # Chuẩn hoá lại: bỏ khoảng trắng dư quanh dấu '/'
    raw = match.group(0)
    return re.sub(r'\s*/\s*', '/', raw).strip()

def search_by_doc_ref(
    doc_ref: str,
    chunk_map: dict,
    faiss_id_map: dict,
    index: "faiss.Index",
    article_index_map: dict,
    dieu_filter: str | None = None,
    khoan_filter: str | None = None,
    content_query: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Tìm các chunk thuộc văn bản có doc_num hoặc title khớp doc_ref.

    Lưu ý về hệ key:
    - chunk_map:        {chunk_id (UUID str): metadata}
    - faiss_id_map:      {faiss_internal_id (int): chunk_id (UUID str)}
    - article_index_map: {"<doc_id>|<article>": [faiss_internal_id, ...]}
      (cùng hệ id số nguyên với faiss_id_map, KHÔNG phải UUID)

    Logic:
    0. Trích số hiệu văn bản thuần từ doc_ref bằng regex (nếu LLM trả kèm
       loại văn bản như 'Nghị định 58/2020/NĐ-CP'); nếu trích được, ưu tiên
       so khớp CHÍNH XÁC theo doc_num đã chuẩn hoá trước khi fallback so khớp mờ
       theo doc_num/title (áp dụng cho trường hợp doc_ref là tên văn bản, ví dụ
       "Luật sở hữu trí tuệ").
    1. Lọc tập chunk_id thuộc văn bản đó -> suy ra doc_id chung.
    2. Nếu có dieu_filter: dùng article_index_map["<doc_id>|<dieu_norm>"] để
       khoanh vùng nhanh theo điều (so khớp chính xác theo doc_id + điều).
       Nếu có thêm khoan_filter: lọc tiếp trong vùng đó theo metadata
       thật (chunk_map[cid]["clause"]).
       Nếu chỉ có khoan_filter (không có dieu_filter): lọc trực tiếp trên
       matched_ids theo metadata "clause".
    3. Nếu có content_query và còn nhiều hơn 1 chunk: Semantic Search để lấy
       Top-k chunk liên quan nhất trong phạm vi đã khoanh vùng ở bước 1-2
       (tránh trả về toàn bộ văn bản gây tràn context).
    """
    extracted_doc_num = _extract_doc_num(doc_ref)
    ref_norm = _normalize_doc_ref(extracted_doc_num or doc_ref)

    # ─────────────────────────────────────────────────────────
    # 1. Lọc chunk_map theo doc_num (dạng "54/2014/QH13") hoặc title (dạng
    #    "Luật sở hữu trí tuệ"). Ưu tiên so khớp chính xác doc_num đã trích.
    # ─────────────────────────────────────────────────────────
    matched_ids: list[str] = []  # các chunk_id (UUID)
    doc_id: str | None = None

    if extracted_doc_num:
        extracted_norm = _normalize_doc_ref(extracted_doc_num)
        for chunk_id, meta in chunk_map.items():
            doc_num_norm = _normalize_doc_ref(meta.get("doc_num", ""))
            if doc_num_norm == extracted_norm:
                matched_ids.append(chunk_id)

    if not matched_ids:
        # Fallback: so khớp mờ 2 chiều, dùng cho trường hợp doc_ref là title
        # (ví dụ "Luật sở hữu trí tuệ") hoặc doc_num không trích được.
        for chunk_id, meta in chunk_map.items():
            doc_num_norm = _normalize_doc_ref(meta.get("doc_num", ""))
            title_norm = _normalize_doc_ref(meta.get("title", ""))

            if (
                ref_norm in doc_num_norm
                or ref_norm in title_norm
                or doc_num_norm in ref_norm
                or title_norm in ref_norm
            ):
                matched_ids.append(chunk_id)

    if not matched_ids:
        return []

    # Suy ra doc_id chung của văn bản (để tra article_index_map)
    doc_id = chunk_map[matched_ids[0]].get("doc_id")

    # ─────────────────────────────────────────────────────────
    # 2. Lọc theo điều / khoản nếu có
    # ─────────────────────────────────────────────────────────
    if dieu_filter:
        dieu_norm = _normalize_doc_ref(dieu_filter)
        article_key = f"{doc_id}|{dieu_filter.strip()}"

        # Ưu tiên dùng article_index_map để khoanh vùng nhanh (so khớp đúng
        # định dạng key "<doc_id>|<article>" như khi build index).
        faiss_ids_for_article = article_index_map.get(article_key)

        if faiss_ids_for_article is not None:
            ids_from_article = {
                faiss_id_map[fid]
                for fid in faiss_ids_for_article
                if fid in faiss_id_map
            }
            filtered_by_dieu = [cid for cid in matched_ids if cid in ids_from_article]
        else:
            # Fallback: tra trực tiếp metadata thật trong chunk_map, để chịu
            # được khác biệt nhỏ về định dạng giữa dieu_filter và key đã lưu.
            filtered_by_dieu = [
                cid
                for cid in matched_ids
                if dieu_norm in _normalize_doc_ref(chunk_map[cid].get("article", ""))
            ]

        if filtered_by_dieu:
            matched_ids = filtered_by_dieu

    if khoan_filter:
        khoan_norm = _normalize_doc_ref(khoan_filter)
        filtered_by_khoan = [
            cid
            for cid in matched_ids
            if khoan_norm in _normalize_doc_ref(chunk_map[cid].get("clause", ""))
        ]
        if filtered_by_khoan:
            matched_ids = filtered_by_khoan

    if not matched_ids:
        return []

    # ─────────────────────────────────────────────────────────
    # 3. Semantic Ranking trong phạm vi đã khoanh vùng (matched_ids)
    #    - Không có dieu/khoan_filter: phạm vi là toàn văn bản.
    #    - Có dieu_filter và/hoặc khoan_filter: phạm vi đã hẹp lại tương ứng.
    # ─────────────────────────────────────────────────────────
    if content_query:
        chunk_to_faiss = {v: k for k, v in faiss_id_map.items()}
        matched_faiss_ids = [
            chunk_to_faiss[cid] for cid in matched_ids if cid in chunk_to_faiss
        ]

        if matched_faiss_ids:
            vec = embed_texts([content_query])
            if vec.size > 0:
                k_search = min(len(faiss_id_map), 200)
                scores, ids = index.search(vec, k_search)

                scored_matches = []
                faiss_id_set = set(matched_faiss_ids)

                for score, fid in zip(scores[0], ids[0]):
                    if float(score) < 0.2:
                        continue
                    if fid in faiss_id_set:
                        cid = faiss_id_map[int(fid)]
                        scored_matches.append((float(score), cid))

                scored_matches.sort(reverse=True)
                matched_ids = [cid for _, cid in scored_matches[:top_k]]

    # ─────────────────────────────────────────────────────────
    # 4. Build results
    # ─────────────────────────────────────────────────────────
    results = []
    for cid in matched_ids[:top_k]:
        meta = chunk_map.get(cid, {})
        results.append(
            {
                "chunk_id": cid,
                "score": 1.0,
                "metadata": meta,
                "source": "doc_ref_search",
            }
        )
    print(
        f"Tìm thấy {len(results)} ở {extracted_doc_num or doc_ref}"
        f"{' - ' + dieu_filter if dieu_filter else ''}"
        f"{' - ' + khoan_filter if khoan_filter else ''}"
    )
    return results

# ─────────────────────────────────────────────────────────────
# Tích hợp tổng hợp: search + rerank
# ─────────────────────────────────────────────────────────────

def search_and_rerank(
    query: str,
    index: faiss.Index,
    faiss_id_map: dict,
    chunk_map: dict,
    initial_k: int = 50,
    final_k: int = 20,
    rerank_threshold: float = 0.3,
) -> list[dict]:
    """Pipeline tìm kiếm tiêu chuẩn: Embed -> FAISS -> Rerank."""
    candidates = faiss_search(query, index, faiss_id_map, chunk_map, top_k=initial_k)
    
    
    
    if not candidates:
        return []
        
    reranked = rerank(query, candidates, top_k=final_k)
    return [c for c in reranked if c.get("rerank_score", 0.0) >= rerank_threshold]