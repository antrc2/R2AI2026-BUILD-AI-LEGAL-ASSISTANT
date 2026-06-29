import faiss
import os
import numpy as np
import json
from typing import List
import re

DATA_DIR           = "data"
FAISS_INDEX_FILE   = os.path.join(DATA_DIR, "faiss.index")
FAISS_ID_MAP_FILE  = os.path.join(DATA_DIR, "faiss_id_map.json")
CHUNK_MAP_FILE     = os.path.join(DATA_DIR, "chunk_map.json")
ARTICLE_MAP_FILE = os.path.join(DATA_DIR,"article_index_map.json")
CHUNKS = os.path.join(DATA_DIR, "chunks.json")

class Search:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.index = self.__load_index()
        self.faiss_id_map = self.__load_faiss_id_map()
        self.chunk_map = self.__load_chunk_map()
        self.article_index_map = self.__load_article_index_map()
        self.chunks_text_map = self.__load_chunks_text_map()
        
    
    def __load_index(self,):
        if not os.path.exists(FAISS_INDEX_FILE):
            raise FileNotFoundError(f"Không tìm thấy file index: {FAISS_INDEX_FILE}")
        idx = faiss.read_index(FAISS_INDEX_FILE)
        print(f"[FAISS] Loaded: {idx.ntotal:,} vectors")
        return idx

    def __load_faiss_id_map(self,):
        if not os.path.exists(FAISS_ID_MAP_FILE):
            raise FileNotFoundError(f"Không tìm thấy file map: {FAISS_ID_MAP_FILE}")
        with open(FAISS_ID_MAP_FILE, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    def __load_chunk_map(self,)-> List[dict]:
        if not os.path.exists(CHUNK_MAP_FILE):
            raise FileNotFoundError(f"Không tìm thấy file map: {CHUNK_MAP_FILE}")
        with open(CHUNK_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    def __load_article_index_map(self,):
        if not os.path.exists(ARTICLE_MAP_FILE):
            raise FileNotFoundError(f"Không tìm thấy file map: {ARTICLE_MAP_FILE}")
        with open(ARTICLE_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    def __load_chunks_text_map(self,):
        chunks_text_map = {}
        with open(CHUNKS,'r',encoding='utf-8') as f:
            all_chunks = json.load(f)
        for c in all_chunks:
            chunks_text_map[c["chunk_id"]] = c.get("embed_text", "")
        return chunks_text_map
    
    def __build_embed_text(self,candidate: dict) -> str:
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
    def semantic_search(
        self,
        vec: np.ndarray,
        top_k: int = 20,
    ) -> list[dict]:
        """Tìm kiếm semantic toàn cục."""

            
        scores, ids = self.index.search(vec, top_k)

        results = []
        for score, faiss_idx in zip(scores[0], ids[0]):
            if self.threshold > float(score):
                continue
            if faiss_idx < 0:
                continue
            chunk_id = self.faiss_id_map.get(int(faiss_idx))
            if chunk_id is None:
                continue
            meta = self.chunk_map.get(chunk_id, {})
            results.append({
                "chunk_id": chunk_id,
                "score":    float(score),
                "metadata": meta,
            })
        return results
    
    def doc_ref_search(
        self,
        vec_query: np.ndarray,
        doc_ref: str | None = None,
        article_filter: str | None = None,
        clause_filter: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        
        extracted_doc_num = self.__extract_doc_num(doc_ref)
        ref_norm = self.__normalize_doc_ref(extracted_doc_num or doc_ref)

        # ─────────────────────────────────────────────────────────
        # 1. Lọc chunk_map theo doc_num (dạng "54/2014/QH13") hoặc title (dạng
        #    "Luật sở hữu trí tuệ"). Ưu tiên so khớp chính xác doc_num đã trích.
        # ─────────────────────────────────────────────────────────
        matched_ids: list[str] = []  # các chunk_id (UUID)
        doc_id: str | None = None

        if extracted_doc_num:
            extracted_norm =self.__normalize_doc_ref(extracted_doc_num)
            for chunk_id, meta in self.chunk_map.items():
                doc_num_norm =self.__normalize_doc_ref(meta.get("doc_num", ""))
                if doc_num_norm == extracted_norm:
                    matched_ids.append(chunk_id)

        if not matched_ids:
            # Fallback: so khớp mờ 2 chiều, dùng cho trường hợp doc_ref là title
            # (ví dụ "Luật sở hữu trí tuệ") hoặc doc_num không trích được.
            for chunk_id, meta in self.chunk_map.items():
                doc_num_norm =self.__normalize_doc_ref(meta.get("doc_num", ""))
                title_norm =self.__normalize_doc_ref(meta.get("title", ""))

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
        doc_id = self.chunk_map[matched_ids[0]].get("doc_id")

        # ─────────────────────────────────────────────────────────
        # 2. Lọc theo điều / khoản nếu có
        # ─────────────────────────────────────────────────────────
        if article_filter:
            dieu_norm =self.__normalize_doc_ref(article_filter)
            article_key = f"{doc_id}|{article_filter.strip()}"

            # Ưu tiên dùng article_index_map để khoanh vùng nhanh (so khớp đúng
            # định dạng key "<doc_id>|<article>" như khi build index).
            faiss_ids_for_article = self.article_index_map.get(article_key)

            if faiss_ids_for_article is not None:
                ids_from_article = {
                    self.faiss_id_map[fid]
                    for fid in faiss_ids_for_article
                    if fid in self.faiss_id_map
                }
                filtered_by_dieu = [cid for cid in matched_ids if cid in ids_from_article]
            else:
                # Fallback: tra trực tiếp metadata thật trong chunk_map, để chịu
                # được khác biệt nhỏ về định dạng giữa article_filter và key đã lưu.
                filtered_by_dieu = [
                    cid
                    for cid in matched_ids
                    if dieu_norm in self.__normalize_doc_ref(self.chunk_map[cid].get("article", ""))
                ]

            if filtered_by_dieu:
                matched_ids = filtered_by_dieu

        if clause_filter:
            khoan_norm =self.__normalize_doc_ref(clause_filter)
            filtered_by_khoan = [
                cid
                for cid in matched_ids
                if khoan_norm in self.__normalize_doc_ref(self.chunk_map[cid].get("clause", ""))
            ]
            if filtered_by_khoan:
                matched_ids = filtered_by_khoan

        if not matched_ids:
            return []

        # ─────────────────────────────────────────────────────────
        # 3. Semantic Ranking trong phạm vi đã khoanh vùng (matched_ids)
        #    - Không có dieu/clause_filter: phạm vi là toàn văn bản.
        #    - Có article_filter và/hoặc clause_filter: phạm vi đã hẹp lại tương ứng.
        # ─────────────────────────────────────────────────────────
        if vec_query and len(matched_ids) > 1:
            chunk_to_faiss = {v: k for k, v in self.faiss_id_map.items()}
            matched_faiss_ids = [
                chunk_to_faiss[cid] for cid in matched_ids if cid in chunk_to_faiss
            ]

            if matched_faiss_ids:
                if vec_query.size > 0:
                    k_search = len(self.faiss_id_map)
                    scores, ids = self.index.search(vec_query, k_search)

                    scored_matches = []
                    faiss_id_set = set(matched_faiss_ids)

                    for score, fid in zip(scores[0], ids[0]):
                        if float(score) > self.threshold:
                            continue
                        if fid in faiss_id_set:
                            cid = self.faiss_id_map[int(fid)]
                            scored_matches.append((float(score), cid))

                    scored_matches.sort(reverse=True)
                    matched_ids = [cid for _, cid in scored_matches[:top_k]]

        # ─────────────────────────────────────────────────────────
        # 4. Build results
        # ─────────────────────────────────────────────────────────
        results = []
        for cid in matched_ids[:top_k]:
            meta = self.chunk_map.get(cid, {})
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
            f"{' - ' + article_filter if article_filter else ''}"
            f"{' - ' + clause_filter if clause_filter else ''}"
        )
        return results
    
    def __normalize_doc_ref(self,text: str) -> str:
        """Chuẩn hoá chuỗi số hiệu / tên văn bản để so sánh mờ."""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def __extract_doc_num(self,text: str) -> str | None:
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