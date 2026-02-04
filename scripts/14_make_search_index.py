#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
import numpy as np
from collections import defaultdict

# ---------- hashing: FNV-1a 32-bit (JS와 동일 구현 필요) ----------
def fnv1a32_bytes(bs: bytes) -> int:
    h = 2166136261
    for b in bs:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def normalize_text(s: str) -> str:
    s = s.lower()
    # 공백/구두점 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def trigrams(text: str):
    # substring 검색을 위해 trigram set 사용 (중복 제거)
    t = normalize_text(text)
    if len(t) < 3:
        return set()
    out = set()
    # 너무 긴 텍스트는 비용 커짐 -> 상한
    # t = t[:2000]
    for i in range(len(t) - 2):
        tri = t[i:i+3]
        out.add(fnv1a32_bytes(tri.encode("utf-8")))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True, help=".../web/public/packed")
    ap.add_argument("--tag", required=True, help="n100000_seed42")
    ap.add_argument("--mod", type=int, default=256)
    args = ap.parse_args()

    packed = Path(args.packed)
    tag = args.tag
    mod = args.mod

    ids_path = packed / f"ids_{tag}.uint32"
    snippets_dir = packed / "snippets"
    out_dir = packed / "search_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.fromfile(ids_path, dtype=np.uint32)
    n = ids.shape[0]

    # postings_shards[shard][hash] = list(point_index)
    postings_shards = [defaultdict(list) for _ in range(mod)]

    # 1) snippet shard별로 한 번만 읽고, 그 shard에 속하는 point indices 처리
    # id shard: id % mod 로 snippet file 결정
    bucket = [[] for _ in range(mod)]
    for i in range(n):
        bucket[int(ids[i] % mod)].append(i)

    for sid in range(mod):
        shard_file = snippets_dir / f"snippets_{tag}_{sid:03d}.json"
        if not shard_file.exists():
            raise FileNotFoundError(shard_file)

        with shard_file.open("r", encoding="utf-8") as f:
            shard_map = json.load(f)  # {"33988213": "...", ...}

        for i in bucket[sid]:
            book_id = int(ids[i])
            txt = shard_map.get(str(book_id))
            if not txt:
                continue
            tri_set = trigrams(txt)

            # 각 trigram-hash를 "인덱스 샤드"로 다시 분배: hash % mod
            for h in tri_set:
                out_shard = h % mod
                postings_shards[out_shard][h].append(i)

        print(f"[{sid:03d}] processed points={len(bucket[sid])} map_size={len(shard_map)}")

    # 2) shard별로 vocab + postings binary 저장
    for shard in range(mod):
        vocab = {}  # hash(str) -> [offset, count]
        flat = []

        # key 순서를 고정 (재현성)
        keys = sorted(postings_shards[shard].keys())
        offset = 0

        for h in keys:
            arr = postings_shards[shard][h]
            # 중복 제거 + 정렬(교집합에 유리)
            arr = sorted(set(arr))
            cnt = len(arr)
            vocab[str(h)] = [offset, cnt]
            flat.extend(arr)
            offset += cnt

        flat_np = np.array(flat, dtype=np.uint32)
        bin_path = out_dir / f"search_tri_{tag}_{shard:03d}.u32"
        json_path = out_dir / f"search_tri_{tag}_{shard:03d}.json"

        flat_np.tofile(bin_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        print(f"[index {shard:03d}] keys={len(keys)} postings={len(flat)}")

    # 3) index meta
    meta = {"tag": tag, "mod": mod, "version": 1, "type": "trigram_fnv1a32"}
    with (out_dir / f"search_index_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("done.")

if __name__ == "__main__":
    main()