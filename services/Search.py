import faiss
import os
import numpy as np
import json
import re
from typing import List, Optional, Dict, Any
from services.Chat import ChatService # Để lấy embedding

DATA_DIR           = "data"
FAISS_INDEX_FILE   = os.path.join(DATA_DIR, "faiss.index")
FAISS_ID_MAP_FILE  = os.path.join(DATA_DIR, "faiss_id_map.json")
CHUNK_MAP_FILE     = os.path.join(DATA_DIR, "chunk_map.json")
ARTICLE_MAP_FILE   = os.path.join(DATA_DIR, "article_index_map.json")
CHUNKS_FILE        = os.path.join(DATA_DIR, "chunks.json")

class SearchService:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.chat_service = ChatService() # Dùng để embed query
        
        # Load dữ liệu tĩnh
        if not os.path.exists(FAISS_INDEX_FILE):
            raise FileNotFoundError(f"Không tìm thấy file index: {FAISS_INDEX_FILE}")
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        
        with open(FAISS_ID_MAP_FILE, "r") as f:
            self.faiss_id_map = {int(k): v for k, v in json.load(f).items()}
            
        with open(CHUNK_MAP_FILE, "r", encoding="utf-8") as f:
            self.chunk_map = json.load(f)
            
        with open(ARTICLE_MAP_FILE, "r", encoding="utf-8") as f:
            self.article_index_map = json.load(f)
            
        # Load chunks text (nếu cần thiết cho việc hiển thị nhanh, otherwise lấy từ chunk_map)
        self.chunks_text_map = {}
        if os.path.exists(CHUNKS_FILE):
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            for c in all_chunks:
                self.chunks_text_map[c["chunk_id"]] = c.get("embed_text", "")

    def _get_chunk_content(self, chunk_id: str) -> str:
        """Lấy nội dung full của chunk."""
        # Ưu tiên lấy từ chunks_text_map, nếu không có thì reconstruct từ metadata hoặc trả về rỗng
        if chunk_id in self.chunks_text_map:
            return self.chunks_text_map[chunk_id]
        
        meta = self.chunk_map.get(chunk_id, {})
        # Fallback: ghép metadata nếu không có text sẵn (tùy cấu trúc data của bạn)
        parts = []
        if meta.get("title"): parts.append(meta["title"])
        if meta.get("article"): parts.append(meta["article"])
        if meta.get("content"): parts.append(meta["content"])
        return " | ".join(parts)

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ nghĩa toàn cục:
        1. Embed query.
        2. FAISS search.
        3. Rerank kết quả.
        """
        # 1. Embedding
        vec = self.chat_service.get_embedding(query)
        if not vec:
            return []
        
        vec_np = np.array([vec], dtype=np.float32)
        
        # 2. FAISS Search
        scores, ids = self.index.search(vec_np, min(top_k * 2, self.index.ntotal)) # Lấy nhiều hơn để rerank
        
        candidates = []
        docs_text_for_rerank = []
        
        for score, faiss_idx in zip(scores[0], ids[0]):
            if faiss_idx < 0: continue
            chunk_id = self.faiss_id_map.get(int(faiss_idx))
            if chunk_id is None: continue
            
            meta = self.chunk_map.get(chunk_id, {})
            content = self._get_chunk_content(chunk_id)
            
            # Filter threshold sơ bộ
            if float(score) < self.threshold:
                continue
                
            candidates.append({
                "chunk_id": chunk_id,
                "faiss_score": float(score),
                "metadata": meta,
                "content": content
            })
            docs_text_for_rerank.append(content)
        
        if not candidates:
            return []
            
        # 3. Rerank
        rerank_scores = self.chat_service.get_rerank_scores(query, docs_text_for_rerank)
        
        # Gán lại score và sort
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = rerank_scores[i]
            candidate["final_score"] = rerank_scores[i] # Ưu tiên rerank score
        
        # Sort theo rerank_score giảm dần
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Cắt top_k cuối cùng
        final_results = []
        for item in candidates[:top_k]:
            final_results.append({
                "chunk_id": item["chunk_id"],
                "score": item["final_score"],
                "metadata": item["metadata"],
                "content": item["content"] # Nội dung đầy đủ để đưa vào context
            })
            
        return final_results

    def doc_ref_search(
        self,
        query: str,
        doc_ref: str,
        article_filter: Optional[str] = None,
        clause_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm trong văn bản cụ thể (dùng cho tool).
        Hỗ trợ lọc theo Số hiệu, Điều, Khoản trước khi Semantic Rank.
        """
        extracted_doc_num = self._extract_doc_num(doc_ref)
        ref_norm = self._normalize_doc_ref(extracted_doc_num or doc_ref)
        
        # 1. Lọc chunk_map theo doc_ref
        matched_ids: List[str] = []
        
        # Ưu tiên match chính xác doc_num
        if extracted_doc_num:
            extracted_norm = self._normalize_doc_ref(extracted_doc_num)
            for chunk_id, meta in self.chunk_map.items():
                doc_num_norm = self._normalize_doc_ref(meta.get("doc_num", ""))
                if doc_num_norm == extracted_norm:
                    matched_ids.append(chunk_id)
        
        # Fallback match mờ nếu không tìm thấy
        if not matched_ids:
            for chunk_id, meta in self.chunk_map.items():
                doc_num_norm = self._normalize_doc_ref(meta.get("doc_num", ""))
                title_norm = self._normalize_doc_ref(meta.get("title", ""))
                if (ref_norm in doc_num_norm or ref_norm in title_norm or 
                    doc_num_norm in ref_norm or title_norm in ref_norm):
                    matched_ids.append(chunk_id)
                    
        if not matched_ids:
            return []
            
        # 2. Lọc theo Điều/Khoản nếu có
        if article_filter:
            dieu_norm = self._normalize_doc_ref(article_filter)
            # Thử match key trong article_index_map trước
            doc_id = self.chunk_map[matched_ids[0]].get("doc_id")
            article_key = f"{doc_id}|{article_filter.strip()}"
            
            faiss_ids_for_article = self.article_index_map.get(article_key)
            if faiss_ids_for_article:
                ids_from_article = {
                    self.faiss_id_map[fid] for fid in faiss_ids_for_article 
                    if fid in self.faiss_id_map
                }
                matched_ids = [cid for cid in matched_ids if cid in ids_from_article]
            else:
                # Fallback scan metadata
                matched_ids = [
                    cid for cid in matched_ids 
                    if dieu_norm in self._normalize_doc_ref(self.chunk_map[cid].get("article", ""))
                ]
        
        if clause_filter:
            khoan_norm = self._normalize_doc_ref(clause_filter)
            matched_ids = [
                cid for cid in matched_ids 
                if khoan_norm in self._normalize_doc_ref(self.chunk_map[cid].get("clause", ""))
            ]
            
        if not matched_ids:
            return []
            
        # 3. Semantic Ranking trong tập đã lọc
        # Nếu chỉ còn 1 ít kết quả sau lọc, có thể trả về luôn hoặc vẫn embed để sort
        if len(matched_ids) > 1:
            vec = self.chat_service.get_embedding(query)
            if vec:
                vec_np = np.array([vec], dtype=np.float32)
                chunk_to_faiss = {v: k for k, v in self.faiss_id_map.items()}
                matched_faiss_ids = [chunk_to_faiss[cid] for cid in matched_ids if cid in chunk_to_faiss]
                
                if matched_faiss_ids:
                    # Search trên toàn index nhưng chỉ quan tâm các id trong matched_faiss_ids
                    # Hoặc tạo temp index (phức tạp), ở đây ta search rộng rồi filter
                    k_search = min(len(matched_faiss_ids) + 5, self.index.ntotal)
                    scores, ids = self.index.search(vec_np, k_search)
                    
                    scored_matches = []
                    faiss_id_set = set(matched_faiss_ids)
                    
                    for score, fid in zip(scores[0], ids[0]):
                        if fid in faiss_id_set and float(score) >= self.threshold:
                            cid = self.faiss_id_map[int(fid)]
                            scored_matches.append((float(score), cid))
                    
                    scored_matches.sort(reverse=True)
                    matched_ids = [cid for _, cid in scored_matches[:top_k]]
        
        # 4. Build result
        results = []
        for cid in matched_ids[:top_k]:
            meta = self.chunk_map.get(cid, {})
            content = self._get_chunk_content(cid)
            results.append({
                "chunk_id": cid,
                "score": 1.0, # Default score cao vì đã filter thủ công
                "metadata": meta,
                "content": content,
                "source": "doc_ref_search"
            })
            
        return results

    def _normalize_doc_ref(self, text: str) -> str:
        if not text: return ""
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _extract_doc_num(self, text: str) -> Optional[str]:
        if not text: return None
        match = re.search(
            r'\d+[A-Za-z]*\s*/\s*\d{4}\s*/\s*[A-ZĐƯƠ]+(?:-[A-ZĐƯƠ]+)*',
            text, flags=re.UNICODE
        )
        if not match: return None
        raw = match.group(0)
        return re.sub(r'\s*/\s*', '/', raw).strip()