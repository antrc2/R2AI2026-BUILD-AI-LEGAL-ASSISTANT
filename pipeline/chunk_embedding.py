"""
PHASE 3 & 4: CHUNKING + EMBEDDING
- Parse HTML từ document content → chunks (Điều/Khoản/Điểm)
- Embed chunks với FAISS
- Resume capability khi tắt
"""

import os
import json
import re
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

EMBEDDING_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-qwen3-embedding-0.6b')
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", 'dont need')

EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 5
FLUSH_EVERY = 500

MAX_TOKENS = 7500
OVERLAP = 1000

DATA_DIR = "data"

INPUT_FILE = "processed_data.json"
CHUNKS_FILE = f"{DATA_DIR}/chunks.json"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss.index"
INDEXED_IDS_FILE = f"{DATA_DIR}/indexed_ids.json"

# Map files cho search
FAISS_ID_MAP_FILE = f"{DATA_DIR}/faiss_id_map.json"
CHUNK_MAP_FILE = f"{DATA_DIR}/chunk_map.json"
DOC_INDEX_MAP_FILE = f"{DATA_DIR}/doc_index_map.json"
ARTICLE_INDEX_MAP_FILE = f"{DATA_DIR}/article_index_map.json"

os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_URL
)


# ─────────────────────────────────────────────────────────────
# PHASE 3: CHUNKING
# ─────────────────────────────────────────────────────────────

def extract_dieu_so(article_full_title: str) -> str:
    """Chỉ lấy phần 'Điều X' từ full title"""
    match = re.match(r'(Điều\s+\d+)', article_full_title)
    return match.group(1) if match else ''


def get_element_text_with_children(el):
    """Lấy toàn bộ text của element bao gồm cả children"""
    return el.get_text(separator=' ', strip=True)


def flatten_metadata(nodes, parent_chain=None):
    """Flatten tree structure sang flat dict"""
    result = {}
    if parent_chain is None:
        parent_chain = []

    for node in nodes:
        node_id = node['id']
        current_chain = parent_chain + [{
            'id': node_id,
            'title': node['title'],
            'level': node['level'],
            'ptype': node['ptype'],
            'orderIndex': node['orderIndex']
        }]

        is_leaf = node.get('isLeaf', False) or 'children' not in node or len(node.get('children', [])) == 0

        result[node_id] = {
            'id': node_id,
            'title': node['title'],
            'level': node['level'],
            'ptype': node['ptype'],
            'orderIndex': node['orderIndex'],
            'isLeaf': is_leaf,
            'ancestors': parent_chain,
            'full_chain': current_chain
        }
        if 'children' in node:
            result.update(flatten_metadata(node['children'], current_chain))

    return result


def parse_document(item) -> list:
    """
    Parse 1 văn bản HTML thành chunks (Điều/Khoản/Điểm).
    
    Return: list of {chunk_id, embed_text, metadata}
    """
    title = item['title']
    doc_num = item['docNum']
    doc_id = str(item['id'])
    
    # Check required fields
    if 'documentContent' not in item or 'content' not in item.get('documentContent', {}):
        return []
    
    html_content = item['documentContent']['content']
    metadata_nodes = item.get('metadata', [])
    
    if not metadata_nodes:
        return []
    
    meta_map = flatten_metadata(metadata_nodes)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Build map: element_id -> BeautifulSoup element
    id_to_el = {}
    for el in soup.find_all(['p', 'div'], id=True):
        el_id = el.get('id')
        if el_id and el_id in meta_map:
            id_to_el[el_id] = el

    # Build map: article_id -> full title text (từ prov-article element)
    node_full_title = {}
    for el in soup.find_all(['p', 'div'], id=True):
        el_id = el.get('id')
        el_class = el.get('class', [])
        if el_id and 'prov-article' in el_class:
            node_full_title[el_id] = el.get_text(separator=' ', strip=True)

    def get_article_ancestor_id(node):
        for ancestor in reversed(node['full_chain']):
            if ancestor['level'] == 'Article':
                return ancestor['id']
        return None

    def get_content_for_leaf(leaf_id, node):
        """Lấy nội dung text cho leaf node"""
        # Trường hợp KHOẢN / ĐIỂM
        if node['level'] != 'Article' and leaf_id in id_to_el:
            el = id_to_el[leaf_id]
            full_text = get_element_text_with_children(el)
            node_title = node['title']
            if full_text.startswith(node_title):
                content = full_text[len(node_title):].strip().lstrip('.:').strip()
            else:
                content = full_text
            return content

        # Trường hợp ĐIỀU không có Khoản
        if leaf_id not in id_to_el:
            return ''

        article_el = id_to_el[leaf_id]
        content_parts = []

        # Duyệt các sibling tiếp theo
        for sibling in article_el.next_siblings:
            if not hasattr(sibling, 'get'):
                continue

            sib_class = sibling.get('class', [])

            # Gặp section tiếp theo → dừng
            if any(c in sib_class for c in (
                'prov-article', 'prov-chapter', 'prov-part',
                'prov-section', 'prov-subsection'
            )):
                break

            # Lấy text từ content elements
            if any(c in sib_class for c in ('prov-content', 'prov-clause', 'prov-item')):
                text = get_element_text_with_children(sibling)
                if text:
                    content_parts.append(text)

        return ' '.join(content_parts)

    def create_leaf_chunk(leaf_id):
        """Tạo 1 chunk từ leaf node"""
        node = meta_map[leaf_id]
        article_id = get_article_ancestor_id(node)

        # Full title của Điều
        art_full_title = node_full_title.get(article_id, '') if article_id else ''
        
        if node['level'] == 'Article':
            art_full_title = node_full_title.get(leaf_id, node['title'])

        # Số điều
        dieu_so = extract_dieu_so(art_full_title) if art_full_title else ''

        # Nội dung text
        own_content = get_content_for_leaf(leaf_id, node)

        SKIP_LEVELS = {'Part', 'Chapter'}

        # Hierarchy string
        hierarchy_titles = [
            n['title'] for n in node['full_chain']
            if n['level'] not in SKIP_LEVELS and n['level'] != 'Article'
        ]
        hierarchy_str = ' - '.join(hierarchy_titles) if hierarchy_titles else ''

        # Build embed_text
        parts = [title]
        if art_full_title:
            parts.append(art_full_title)
        if hierarchy_str:
            parts.append(hierarchy_str)

        embed_text = ' - '.join(parts)
        if own_content:
            embed_text += f': {own_content}'

        # Metadata
        metadata = {
            'title': title,
            'doc_num': doc_num,
            'doc_id': doc_id,
        }
        
        if dieu_so:
            metadata['article'] = dieu_so
        
        for n in node['full_chain']:
            if n['level'] == 'Clause':
                metadata['clause'] = n['title']
                break

        return {
            'chunk_id': leaf_id,
            'embed_text': embed_text,
            'metadata': metadata
        }

    chunks = []
    for el in soup.find_all(['p', 'div'], id=True):
        el_id = el.get('id')
        if not el_id or el_id not in meta_map:
            continue
        node_info = meta_map[el_id]
        if node_info['isLeaf']:
            chunks.append(create_leaf_chunk(el_id))

    return chunks


def phase3_chunk():
    """Tách document thành chunks"""
    print("=" * 80)
    print("PHASE 3: CHUNKING (HTML → Chunks)")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Đọc {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File {INPUT_FILE} không tồn tại! Chạy phase 1+2 trước.")
        return []
    
    print(f"   Tổng docs: {len(data)}")
    
    # Build chunks
    print(f"\n2. Parse HTML → chunks...")
    
    all_chunks = []
    skipped = []
    
    for idx, item in enumerate(data):
        doc_num = item.get('docNum', 'N/A')
        try:
            if 'metadata' not in item:
                raise KeyError("Thiếu field 'metadata'")
            if 'documentContent' not in item:
                raise KeyError("Thiếu field 'documentContent'")
            if 'content' not in item.get('documentContent', {}):
                raise KeyError("Thiếu field 'documentContent.content'")

            chunks = parse_document(item)
            all_chunks.extend(chunks)
            
            if idx % 100 == 0:
                print(f"   [{idx+1}/{len(data)}] {doc_num} → {len(chunks)} chunks")

        except Exception as e:
            skipped.append({
                'doc_num': doc_num,
                'id': item.get('id', 'N/A'),
                'title': item.get('title', 'N/A'),
                'error': str(e)
            })
    
    print(f"   ✓ {len(all_chunks)} chunks từ {len(data)} docs")
    if skipped:
        print(f"   ✗ {len(skipped)} docs skip")
    
    # Build index maps
    print(f"\n3. Build index maps...")
    
    faiss_id_map = {}
    chunk_map = {}
    doc_index_map = {}
    article_index_map = {}
    
    for faiss_idx, chunk in enumerate(all_chunks):
        chunk_id = chunk['chunk_id']
        meta = chunk['metadata']
        doc_id = meta['doc_id']
        article = meta.get('article', '')

        faiss_id_map[faiss_idx] = chunk_id
        chunk_map[chunk_id] = meta

        doc_index_map.setdefault(doc_id, []).append(faiss_idx)

        if article:
            key = f"{doc_id}|{article}"
            article_index_map.setdefault(key, []).append(faiss_idx)
    
    # Save
    print(f"\n4. Lưu chunks...")
    
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {CHUNKS_FILE} ({len(all_chunks)} chunks)")
    
    with open(FAISS_ID_MAP_FILE, 'w') as f:
        json.dump(faiss_id_map, f)
    print(f"   ✓ {FAISS_ID_MAP_FILE}")
    
    with open(CHUNK_MAP_FILE, 'w',encoding='utf-8') as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {CHUNK_MAP_FILE}")
    
    with open(DOC_INDEX_MAP_FILE, 'w') as f:
        json.dump(doc_index_map, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {DOC_INDEX_MAP_FILE}")
    
    with open(ARTICLE_INDEX_MAP_FILE, 'w',encoding='utf-8') as f:
        json.dump(article_index_map, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {ARTICLE_INDEX_MAP_FILE}")
    
    if skipped:
        with open(f"{DATA_DIR}/skipped.json", 'w', encoding='utf-8') as f:
            json.dump(skipped, f, ensure_ascii=False, indent=2)
        print(f"   ✓ skipped.json ({len(skipped)} errors)")
    
    print(f"\n✅ Chunking hoàn tất!")
    return all_chunks


# ─────────────────────────────────────────────────────────────
# PHASE 4: EMBEDDING
# ─────────────────────────────────────────────────────────────

def normalize(vectors):
    """Normalize vectors"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def embed_texts(texts):
    """Embed batch texts"""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    data = sorted(resp.data, key=lambda x: x.index)
    vecs = np.array([x.embedding for x in data], dtype=np.float32)

    return normalize(vecs)


def safe_embed(text):
    """Embed 1 text, handle errors"""
    try:
        vec = embed_texts([text])
        return vec[0]

    except Exception as e:
        msg = str(e)
        if "maximum context length" not in msg and "input_tokens" not in msg:
            raise

        print("[truncate] Text quá dài, skip chunk")
        return None


def phase4_embedding(all_chunks=None):
    """Embed chunks"""
    print("\n" + "=" * 80)
    print("PHASE 4: EMBEDDING (Chunks → FAISS)")
    print("=" * 80)
    
    # Load chunks
    print(f"\n1. Đọc chunks...")
    
    if all_chunks is None:
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
        except FileNotFoundError:
            print(f"❌ File {CHUNKS_FILE} không tồn tại! Chạy phase 3 trước.")
            return
    
    print(f"   Tổng chunks: {len(all_chunks)}")
    
    # Load hoặc tạo FAISS index
    print(f"\n2. Load/tạo FAISS index...")
    
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"   ✓ Resume index: {index.ntotal} vectors")
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        print(f"   ✓ Tạo index mới")
    
    # Load indexed IDs
    if os.path.exists(INDEXED_IDS_FILE):
        with open(INDEXED_IDS_FILE, "r") as f:
            indexed_ids = set(json.load(f))
    else:
        indexed_ids = set()
    
    print(f"   Đã indexed: {len(indexed_ids)}")
    
    # Embed
    print(f"\n3. Embedding chunks...")
    
    pending_vectors = []
    pending_ids = []
    flush_counter = 0
    skipped_count = 0
    
    for chunk_idx, chunk in enumerate(all_chunks):
        
        if chunk_idx in indexed_ids:
            continue

        text = chunk["embed_text"]

        try:
            vec = safe_embed(text)

            if vec is None:
                skipped_count += 1
                continue

            pending_vectors.append(vec)
            pending_ids.append(chunk_idx)

        except Exception as e:
            print(f"Chunk {chunk_idx} failed: {e}")
            skipped_count += 1
            continue

        # Flush
        if len(pending_vectors) >= EMBED_BATCH_SIZE:
            vecs = np.array(pending_vectors, dtype=np.float32)
            ids = np.array(pending_ids, dtype=np.int64)

            index.add_with_ids(vecs, ids)
            indexed_ids.update(pending_ids)
            flush_counter += len(pending_ids)

            pending_vectors.clear()
            pending_ids.clear()

            if flush_counter >= FLUSH_EVERY:
                faiss.write_index(index, FAISS_INDEX_FILE)
                with open(INDEXED_IDS_FILE, "w") as f:
                    json.dump(sorted(indexed_ids), f)
                
                print(f"   💾 Saved: {index.ntotal} vectors (skip: {skipped_count})")
                flush_counter = 0
    
    # Final flush
    if pending_vectors:
        vecs = np.array(pending_vectors, dtype=np.float32)
        ids = np.array(pending_ids, dtype=np.int64)
        index.add_with_ids(vecs, ids)
        indexed_ids.update(pending_ids)
    
    # Save final
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(INDEXED_IDS_FILE, "w") as f:
        json.dump(sorted(indexed_ids), f)
    
    print(f"\n✅ Embedding hoàn tất!")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Skipped: {skipped_count}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "phase4":
        # Chỉ embedding
        phase4_embedding()
    elif len(sys.argv) > 1 and sys.argv[1] == "phase3":
        # Chỉ chunking
        phase3_chunk()
    else:
        # Cả 2 phase
        chunks = phase3_chunk()
        phase4_embedding(chunks)
