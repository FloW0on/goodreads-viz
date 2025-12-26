#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute sentence embeddings for sampled Goodreads texts and save as .npy + metadata."
    )
    p.add_argument("--input", required=True, help="Parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Embedding compute dtype (float16 saves GPU memory).",
    )
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (good for cosine search).")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--tag", default="", help="Suffix tag for output filenames")
    return p.parse_args()


def make_out_paths(out_dir: Path, tag: str):
    tag = tag.strip()
    suffix = f"_{tag}" if tag else ""
    emb_path = out_dir / f"embeddings{suffix}.npy"
    ids_path = out_dir / f"ids{suffix}.npy"
    meta_path = out_dir / f"embed_meta{suffix}.json"
    return emb_path, ids_path, meta_path


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to fail fast with clear messages if missing
    import torch  # noqa
    from sentence_transformers import SentenceTransformer  # noqa

    print(f"Loading input parquet: {in_path}")
    df = pd.read_parquet(in_path)

    if "id" not in df.columns or "text_for_embed" not in df.columns:
        raise ValueError("Input parquet must contain columns: 'id', 'text_for_embed'")

    # Clean minimal (avoid None)
    df["text_for_embed"] = df["text_for_embed"].fillna("").astype(str)
    df = df[df["text_for_embed"].str.len() > 0].copy()

    ids = df["id"].astype(np.int64).to_numpy()
    texts = df["text_for_embed"].tolist()

    n = len(texts)
    print(f"Records to embed: {n:,}")

    # Load model
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length

    # Decide dtype
    use_fp16 = (args.dtype == "float16") and (args.device.startswith("cuda"))
    if use_fp16:
        # SentenceTransformers uses torch under the hood; this helps reduce memory.
        # Some models may not support fp16 perfectly, but MiniLM does fine.
        model = model.half()
        print("Using fp16 (model.half())")
    else:
        print("Using fp32")

    # Embed in batches
    all_embs = []
    bs = args.batch_size
    num_batches = math.ceil(n / bs)

    print(f"Embedding with batch_size={bs}, max_seq_length={args.max_seq_length}, device={args.device}")
    for b in tqdm(range(num_batches), desc="embedding"):
        s = b * bs
        e = min((b + 1) * bs, n)
        batch_texts = texts[s:e]

        # encode returns numpy by default if convert_to_numpy=True
        embs = model.encode(
            batch_texts,
            batch_size=bs,  # SBERT internal batching; same as outer, ok
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we handle optional normalize below
        )
        all_embs.append(embs)

    embs = np.vstack(all_embs).astype(np.float32)  # store as float32 for downstream stability
    print(f"Embeddings shape: {embs.shape}")

    if args.normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        print("Applied L2 normalization")

    emb_path, ids_path, meta_path = make_out_paths(out_dir, args.tag)

    # Save
    np.save(emb_path, embs)
    np.save(ids_path, ids)

    meta = {
        "input": str(in_path),
        "n_records": int(n),
        "embedding_dim": int(embs.shape[1]),
        "model": args.model,
        "batch_size": int(args.batch_size),
        "max_seq_length": int(args.max_seq_length),
        "dtype_compute": args.dtype,
        "stored_dtype": "float32",
        "normalize": bool(args.normalize),
        "device": args.device,
        "tag": args.tag,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", emb_path)
    print(" -", ids_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()