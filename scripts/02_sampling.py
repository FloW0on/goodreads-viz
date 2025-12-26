#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gzip
import html
import json
import random
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


TAG_RE = re.compile(r"<[^>]+>")


def clean_description(s: str, max_chars: int = 800) -> str:
    """Clean Goodreads HTML description."""
    if not s:
        return ""
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def build_text_for_embed(title: str, desc: str) -> str:
    title = (title or "").strip()
    desc = (desc or "").strip()
    if title and desc:
        return f"{title}\n\n{desc}"
    return title or desc


def parse_args():
    p = argparse.ArgumentParser(
        description="Reservoir-sample English books with description from goodreads_books.json.gz"
    )
    p.add_argument("--input", required=True, help="Path to goodreads_books.json.gz")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--sample_size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_desc_chars", type=int, default=800)
    p.add_argument("--progress_every", type=int, default=100_000)
    p.add_argument("--lang_codes", nargs="*", default=["en", "eng", "en-US", "en-GB"])
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_codes = set(args.lang_codes)

    reservoir = []
    seen = 0

    print("Streaming reservoir sampling...")
    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)

            raw_desc = (book.get("description") or "")
            has_desc = bool(raw_desc.strip())
            is_en = (book.get("language_code") or "").strip() in lang_codes
            if not (has_desc and is_en):
                continue

            seen += 1

            if len(reservoir) < args.sample_size:
                reservoir.append(book)
            else:
                j = random.randint(0, seen - 1)
                if j < args.sample_size:
                    reservoir[j] = book

            if args.progress_every and (seen % args.progress_every == 0):
                print(f"  eligible seen: {seen:,}")

    print(f"Done. Sample size: {len(reservoir):,} (eligible seen: {seen:,})")

    meta_rows = []
    text_rows = []

    for b in reservoir:
        bid_raw = str(b.get("book_id") or "").strip()
        if not bid_raw.isdigit():
            continue
        book_id = int(bid_raw)

        title = (b.get("title_without_series") or b.get("title") or "").strip()
        desc = clean_description(b.get("description") or "", max_chars=args.max_desc_chars)
        text_for_embed = build_text_for_embed(title, desc)

        meta_rows.append(
            {
                "id": book_id,
                "title": title,
                "language_code": (b.get("language_code") or "").strip(),
                "average_rating": float(b.get("average_rating") or "nan"),
                "ratings_count": int(b.get("ratings_count") or 0),
                "url": (b.get("url") or b.get("link") or "").strip(),
                "image_url": (b.get("image_url") or "").strip(),
                "num_pages": int(b.get("num_pages") or 0)
                if str(b.get("num_pages") or "").isdigit()
                else None,
                "publication_year": int(b.get("publication_year") or 0)
                if str(b.get("publication_year") or "").isdigit()
                else None,
            }
        )
        text_rows.append(
            {
                "id": book_id,
                "text_for_embed": text_for_embed,
            }
        )

    meta_df = pd.DataFrame(meta_rows).drop_duplicates("id")
    text_df = pd.DataFrame(text_rows).drop_duplicates("id")

    tag = f"n{args.sample_size}_seed{args.seed}"
    meta_path = out_dir / f"sample_meta_{tag}.parquet"
    text_path = out_dir / f"sample_texts_{tag}.parquet"
    ids_path = out_dir / f"sample_ids_{tag}.txt"

    meta_df.to_parquet(meta_path, index=False)
    text_df.to_parquet(text_path, index=False)
    ids_path.write_text("\n".join(map(str, meta_df["id"].tolist())), encoding="utf-8")

    print("Saved:")
    print(" -", meta_path)
    print(" -", text_path)
    print(" -", ids_path)


if __name__ == "__main__":
    main()