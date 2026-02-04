import json
from pathlib import Path

DATA_PATH = Path(r"C:\book_atlas\dataset\raw\goodreads_books.json")

def try_parse_as_json_array():
    # 배열 JSON인지만 확인하려는 목적이라 1MB 정도만 읽어서 판단
    with DATA_PATH.open("rb") as f:
        head = f.read(1024 * 1024).lstrip()
    return head[:1] == b"["

def probe_ndjson(n_lines=5):
    rows = []
    with DATA_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Not found: {DATA_PATH}")

    is_array = try_parse_as_json_array()
    print(f"[probe] starts with '[' ? {is_array}  (True면 배열 JSON 가능성)")

    # NDJSON 샘플 파싱 시도
    try:
        rows = probe_ndjson(5)
        print(f"[probe] NDJSON parse OK: {len(rows)} records from first lines")
        if rows:
            keys = sorted(rows[0].keys())
            print(f"[probe] sample keys (first record): {keys}")
            for k in ["book_id", "title", "description", "popular_shelves", "authors", "average_rating", "ratings_count"]:
                print(f"  - has '{k}': {k in rows[0]}")
    except Exception as e:
        print("[probe] NDJSON parse FAILED:", repr(e))
        print("배열 JSON이거나, 라인 단위가 아닌 다른 포맷 가능성.")

if __name__ == "__main__":
    main()