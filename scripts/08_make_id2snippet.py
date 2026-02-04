#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser("Make sharded id->text snippet json files + index")
    p.add_argument("--texts", required=True, help="Parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory (e.g., web/public/packed)")
    p.add_argument("--tag", required=True, help="Tag like n100000_seed42")
    p.add_argument("--max_chars", type=int, default=160)
    p.add_argument("--shards", type=int, default=256, help="Number of shard files (recommended: 256 or 512)")
    p.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Shard file format. json = {id: snippet}. jsonl = one record per line.",
    )
    return p.parse_args()

def clean_snippet(text: str, max_chars: int) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text

def shard_id(book_id: int, shards: int) -> int:
    return int(book_id) % int(shards)

def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.texts, columns=["id", "text_for_embed"])
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64")

    # Shard buffers in memory
    buckets = defaultdict(list)

    for r in df.itertuples(index=False):
        bid = int(r.id)
        snip = clean_snippet(r.text_for_embed, a.max_chars)
        if not snip:
            continue
        s = shard_id(bid, a.shards)
        buckets[s].append((bid, snip))

    total = sum(len(v) for v in buckets.values())

    # Write shards
    shard_files = {}
    for s in range(a.shards):
        items = buckets.get(s, [])
        shard_name = f"snippets_{a.tag}_{s:03d}.{ 'jsonl' if a.format=='jsonl' else 'json' }"
        shard_path = out_dir / shard_name

        if a.format == "json":
            obj = {str(bid): snip for bid, snip in items}
            shard_path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        else:
            with shard_path.open("w", encoding="utf-8") as f:
                for bid, snip in items:
                    f.write(json.dumps({"id": bid, "snippet": snip}, ensure_ascii=False) + "\n")

        shard_files[str(s)] = shard_name

    # Write index
    index = {
        "tag": a.tag,
        "scheme": "mod",
        "mod": int(a.shards),
        "max_chars": int(a.max_chars),
        "format": a.format,
        "pattern": f"snippets_{a.tag}" + "_{shard:03d}." + ("jsonl" if a.format == "jsonl" else "json"),
        "files": shard_files,
        "n_records": int(total),
    }
    index_path = out_dir / f"snippets_index_{a.tag}.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved shards to:", out_dir)
    print("Index:", index_path)
    print("Total snippets:", total)

if __name__ == "__main__":
    main()