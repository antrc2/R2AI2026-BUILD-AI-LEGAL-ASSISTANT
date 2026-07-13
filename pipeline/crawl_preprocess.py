"""
PHASE 1 & 2: CRAWL + PREPROCESS
- Crawl từ VBPL API (với checkpoint continue)
- Lọc dữ liệu (effStatus hợp lệ)
- Merge dữ liệu cleaned + index → processed_data.json sẵn sàng
"""

import requests
import json
import time
import os
from datetime import datetime
from slugify import slugify
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CHECKPOINT_FILE = "checkpoint.json"
CRAWL_DATA_FILE = "data.jsonl"
CRAWL_INDEX_FILE = "index_data.jsonl"
PROCESSED_DATA_FILE = "processed_data.json"


TOTAL_PAGES = 36916
MAX_PAGE_RETRIES = 5
MAX_DOC_RETRIES = 3


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """Session với retry tự động"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ─────────────────────────────────────────────────────────────
# Headers
# ─────────────────────────────────────────────────────────────

def get_index_headers(doc_id: str, title_slug: str) -> dict:
    """Headers cho index API"""
    url_id = f"{title_slug}--{doc_id}"
    return {
        "accept": "text/x-component",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "text/plain;charset=UTF-8",
        "next-action": "94635012466e8fede44782d4237c10fe75501920",
        "next-router-state-tree": f'%5B%22%22%2C%7B%22children%22%3A%5B%22van-ban%22%2C%7B%22children%22%3A%5B%5B%22category%22%2C%22chi-tiet%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%5B%22id%22%2C%22{url_id}%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%2Ctrue%5D',
        "pragma": "no-cache",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load checkpoint hoặc tạo mới"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_page": 0, "failed_docs": []}


def save_checkpoint(last_page: int, failed_docs: list):
    """Lưu checkpoint"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_page": last_page, "failed_docs": failed_docs}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Process Doc
# ─────────────────────────────────────────────────────────────

def process_doc(session: requests.Session, doc_id: str, data_f, index_f) -> bool:
    """Lấy detail + index data cho 1 doc"""
    for attempt in range(1, MAX_DOC_RETRIES + 1):
        try:
            # 1. Lấy detail
            detail_response = session.get(
                f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}",
                timeout=30,
            )
            detail_response.raise_for_status()
            data_detail = detail_response.json().get("data", {})
            data_f.write(json.dumps(data_detail, ensure_ascii=False) + "\n")

            # 2. Lấy index
            title_slug = slugify(data_detail.get("title", ""))
            index_url = f"https://vbpl.vn/van-ban/chi-tiet/{title_slug}--{doc_id}"

            index_response = session.post(
                url=index_url,
                headers=get_index_headers(doc_id, title_slug),
                data=f'["{doc_id}"]',
                timeout=30,
            )
            index_response.encoding = "utf-8"
            index_response.raise_for_status()

            for line in index_response.text.splitlines():
                if line.startswith("1:"):
                    index_data = json.loads(line[2:])
                    index_f.write(
                        json.dumps(
                            {"doc_id": doc_id, "index_data": index_data},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    break

            return True

        except Exception as e:
            wait = 2 ** attempt
            print(f"  [doc {doc_id}] lỗi lần {attempt}/{MAX_DOC_RETRIES}: {e} — chờ {wait}s")
            time.sleep(wait)

    print(f"  [doc {doc_id}] bỏ qua sau {MAX_DOC_RETRIES} lần thất bại")
    return False


# ─────────────────────────────────────────────────────────────
# PHASE 1: CRAWL
# ─────────────────────────────────────────────────────────────

def phase1_crawl():
    """Crawl tất cả dữ liệu từ API"""
    print("=" * 80)
    print("PHASE 1: CRAWL DATA")
    print("=" * 80)

    checkpoint = load_checkpoint()
    start_page = checkpoint["last_page"]
    failed_docs = checkpoint["failed_docs"]

    print(f"Bắt đầu từ trang {start_page + 1}/{TOTAL_PAGES} | Docs lỗi: {len(failed_docs)}")

    session = make_session()

    with open(CRAWL_DATA_FILE, "a", encoding="utf-8") as data_f, \
         open(CRAWL_INDEX_FILE, "a", encoding="utf-8") as index_f:

        for i in range(start_page, TOTAL_PAGES):
            page_num = i + 1

            # --- Retry page ---
            items = None
            for attempt in range(1, MAX_PAGE_RETRIES + 1):
                try:
                    response = session.post(
                        "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all",
                        json={
                            "pageSize": 10,
                            "sortDirection": "desc",
                            "sortBy": "viewCount",
                            "sortByViewCount": True,
                            "pageNumber": page_num,
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    items = response.json().get("data", {}).get("items", [])
                    break

                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[page {page_num}] lỗi lần {attempt}/{MAX_PAGE_RETRIES}: {e} — chờ {wait}s")
                    time.sleep(wait)

            if items is None:
                print(f"[page {page_num}] bỏ qua cả trang sau {MAX_PAGE_RETRIES} lần thất bại")
                save_checkpoint(i, failed_docs)
                continue

            # --- Xử lý từng doc ---
            for item in items:
                doc_id = item.get("id")
                if not doc_id:
                    continue

                success = process_doc(session, doc_id, data_f, index_f)
                if not success:
                    failed_docs.append(doc_id)

            # --- Checkpoint sau mỗi page ---
            data_f.flush()
            index_f.flush()
            save_checkpoint(page_num, failed_docs)
            print(f"✓ Page {page_num}/{TOTAL_PAGES} | Docs lỗi: {len(failed_docs)}")

        # --- Retry failed docs ---
        if failed_docs:
            print(f"\n⚠️  Retry {len(failed_docs)} docs lỗi...")
            still_failed = []
            for doc_id in failed_docs:
                success = process_doc(session, doc_id, data_f, index_f)
                if not success:
                    still_failed.append(doc_id)
                data_f.flush()
                index_f.flush()

            save_checkpoint(TOTAL_PAGES, still_failed)
            if still_failed:
                print(f"Vẫn còn {len(still_failed)} docs lỗi, xem checkpoint.json")
            else:
                print("✅ Tất cả docs đã xử lý xong!")
        else:
            print("\n✅ Hoàn tất, không có doc nào lỗi!")


# ─────────────────────────────────────────────────────────────
# PHASE 2: PREPROCESS & FILTER
# ─────────────────────────────────────────────────────────────

def is_valid_document(item: dict, index_data_dict: dict) -> tuple:
    """
    Kiểm tra xem doc có hợp lệ không.
    
    Rules:
    1. Phải có effStatus hợp lệ
    2. Chỉ chấp nhận: "Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"
    3. Phải có index metadata
    
    Return: (is_valid, reason)
    """
    doc_id = str(item.get("id", ""))
    
    # Check effStatus
    if "effStatus" not in item:
        return False, "missing_effStatus"
    
    if item["effStatus"] is None:
        return False, "effStatus_is_None"
    
    if not isinstance(item["effStatus"], dict):
        return False, "effStatus_not_dict"
    
    if "name" not in item["effStatus"]:
        return False, "effStatus_missing_name"
    
    eff_status_name = item["effStatus"]["name"]
    
    # KEEP: "Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"
    # REMOVE: "Hết hiệu lực toàn bộ", "Không còn phù hợp", "Ngưng hiệu lực"
    if eff_status_name not in {"Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"}:
        return False, f"unsupported_effStatus: {eff_status_name}"
    
    # Check index metadata
    if doc_id not in index_data_dict or not index_data_dict[doc_id]:
        return False, "no_index_metadata"
    
    return True, "valid"


def phase2_preprocess():
    """Filter & merge dữ liệu"""
    print("\n" + "=" * 80)
    print("PHASE 2: PREPROCESS & FILTER")
    print("=" * 80)
    
    # Load crawled data
    print(f"\n1. Đọc {CRAWL_DATA_FILE}...")
    try:
        with open(CRAWL_DATA_FILE, "r", encoding="utf-8") as f:
            all_items = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ File {CRAWL_DATA_FILE} không tồn tại! Chạy phase 1 trước.")
        return
    
    print(f"   Tổng docs crawled: {len(all_items)}")
    
    # Load index data
    print(f"\n2. Đọc {CRAWL_INDEX_FILE}...")
    try:
        with open(CRAWL_INDEX_FILE, "r", encoding="utf-8") as f:
            index_data_list = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ File {CRAWL_INDEX_FILE} không tồn tại! Chạy phase 1 trước.")
        return
    
    # Build index data map
    index_data_dict = {}
    for entry in index_data_list:
        doc_id = entry.get("doc_id")
        index_metadata = entry.get("index_data", [])
        index_data_dict[doc_id] = index_metadata
    
    print(f"   Tổng index entries: {len(index_data_dict)}")
    
    # Filter & merge
    print(f"\n3. Lọc & merge dữ liệu...")
    
    cleaned_data = []
    error_items = []
    
    for item in all_items:
        doc_num = item.get("docNum", "N/A")
        doc_id = str(item.get("id", ""))
        
        is_valid, reason = is_valid_document(item, index_data_dict)
        
        if not is_valid:
            error_items.append({
                "docNum": doc_num,
                "doc_id": doc_id,
                "title": item.get("title"),
                "effStatus": item.get("effStatus"),
                "effFrom": item.get("effFrom"),
                "reason": reason
            })
            continue
        
        # Add index metadata
        item["metadata"] = index_data_dict.get(doc_id, [])
        cleaned_data.append(item)
    
    print(f"   ✓ Docs hợp lệ: {len(cleaned_data)}")
    print(f"   ✗ Docs lỗi: {len(error_items)}")
    
    # Save results
    print(f"\n4. Lưu kết quả...")
    
    with open(PROCESSED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"   ✓ {PROCESSED_DATA_FILE} ({len(cleaned_data)} docs)")
    
    with open("error_items.json", "w", encoding="utf-8") as f:
        json.dump(error_items, f, ensure_ascii=False, indent=2)
    print(f"   ✓ error_items.json ({len(error_items)} errors)")
    
    # Summary by reason
    print(f"\n5. Breakdown lỗi:")
    error_reasons = {}
    for err in error_items:
        reason = err["reason"]
        error_reasons[reason] = error_reasons.get(reason, 0) + 1
    
    for reason, count in sorted(error_reasons.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")
    
    print(f"\n✅ Hoàn tất! Ready cho phase 3 (embedding).")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        # Chỉ chạy phase 2
        phase2_preprocess()
    else:
        # Chạy cả 2 phase
        phase1_crawl()
        phase2_preprocess()
