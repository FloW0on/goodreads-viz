#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


KEEP_COLS = [
    "id",
    "title",
    "average_rating",
    "ratings_count",
    "publication_year",
    "num_pages",
    "url",
    "image_url",
    "language_code",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Export lightweight web metadata jsonl from packed debug parquet."
    )
    p.add_argument("--debug_parquet", required=True, help="points_debug_*.parquet path")
    p.add_argument("--out_dir", required=True, help="packed output directory")
    p.add_argument("--tag", required=True, help="e.g., n10000_seed42")
    return p.parse_args()


def main():
    args = parse_args()
    debug_path = Path(args.debug_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(debug_path)

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in debug parquet: {missing}")

    out_path = out_dir / f"meta_{args.tag}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in df[KEEP_COLS].itertuples(index=False):
            obj = {
                "id": int(row.id),
                "title": (row.title or ""),
                "average_rating": None if pd.isna(row.average_rating) else float(row.average_rating),
                "ratings_count": int(row.ratings_count) if not pd.isna(row.ratings_count) else 0,
                "publication_year": None if pd.isna(row.publication_year) else int(row.publication_year),
                "num_pages": None if pd.isna(row.num_pages) else int(row.num_pages),
                "url": (row.url or ""),
                "image_url": (row.image_url or ""),
                "language_code": (row.language_code or ""),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Saved:", out_path)
    print("Lines:", sum(1 for _ in out_path.open("r", encoding="utf-8")))


if __name__ == "__main__":
    main()