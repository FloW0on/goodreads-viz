#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import html
import json
import random
import re
from pathlib import Path

import pandas as pd

TAG_RE = re.compile(r"<[^>]+>")

# 로컬 고정 경로
RAW_JSON = Path(r"C:\book_atlas\dataset\raw\goodreads_books.json")
OUT_DIR  = Path(r"C:\book_atlas\dataset\processed")

DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_LANG_CODES = {"en", "eng", "en-US", "en-GB"}

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
    p.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_desc_chars", type=int, default=800)
    p.add_argument("--progress_every", type=int, default=200_000)
    p.add_argument("--lang_codes", nargs="*", default=DEFAULT_LANG_CODES)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if not RAW_JSON.exists():
        raise FileNotFoundError(f"Not found: {RAW_JSON}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lang_codes = set(args.lang_codes)

    reservoir = []
    seen = 0

    print(f"Streaming reservoir sampling from: {RAW_JSON}")
    with RAW_JSON.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                book = json.loads(line)
            except Exception:
                continue

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
                print(f"  eligible seen: {seen:,} | resevoir: {len(reservoir):,}")

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

        # average_rating이 빈 문자열일 수 있어 예외 처리
        try:
            avg_rating = float(b.get("average_rating")) if b.get("average_rating") not in (None, "") else float("nan")
        except Exception:
            avg_rating = float("nan")

        # ratings_count도 문자열/None 섞임 방지
        try:
            ratings_count = int(str(b.get("ratings_count") or "0").replace(",", ""))
        except Exception:
            ratings_count = 0

        num_pages = None
        np_raw = b.get("num_pages")
        if np_raw is not None and str(np_raw).strip().isdigit():
            num_pages = int(str(np_raw).strip())

        pub_year = None
        py_raw = b.get("publication_year")
        if py_raw is not None and str(py_raw).strip().isdigit():
            pub_year = int(str(py_raw).strip())

        meta_rows.append(
            {
                "id": book_id,
                "title": title,
                "language_code": (b.get("language_code") or "").strip(),
                "average_rating": avg_rating,
                "ratings_count": ratings_count,
                "url": (b.get("url") or b.get("link") or "").strip(),
                "image_url": (b.get("image_url") or "").strip(),
                "num_pages": num_pages,
                "publication_year": pub_year,
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
    meta_path = OUT_DIR / f"sample_meta_{tag}.parquet"
    text_path = OUT_DIR / f"sample_texts_{tag}.parquet"
    ids_path = OUT_DIR / f"sample_ids_{tag}.txt"

    meta_df.to_parquet(meta_path, index=False)
    text_df.to_parquet(text_path, index=False)
    ids_path.write_text("\n".join(map(str, meta_df["id"].tolist())), encoding="utf-8")

    print("Saved:")
    print(" -", meta_path)
    print(" -", text_path)
    print(" -", ids_path)


if __name__ == "__main__":
    main()