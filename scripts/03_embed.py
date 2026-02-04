#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_embed.py (CPU-optimized)
- Input: Parquet with columns: id, text_for_embed
- Output:
  1) embeddings_*.npy  (float32, memmap write)
  2) ids_*.npy         (int64)
  3) embed_index_*.parquet  (id -> row mapping + meta fields)
  4) embed_meta_*.json      (run metadata)

Design goals:
- CPU-friendly (multi-process encoding supported by SentenceTransformers)
- Avoid holding entire embeddings in RAM (memmap streaming write)
- Keep reproducibility + traceability (parquet + json meta)
"""

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute sentence embeddings (CPU) and save as .npy + index parquet + meta json."
    )
    p.add_argument("--input", required=True, help="Parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory")

    # Model / encoding
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu", help="cpu (recommended here)")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size per worker")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (cosine-friendly)")

    # CPU parallelism
    p.add_argument("--num_workers", type=int, default=0,
                   help="SentenceTransformers multi-process workers (0=auto). Recommended: physical cores-1.")
    p.add_argument("--chunk_size", type=int, default=20000,
                   help="How many rows to feed per encode call (controls memory/overhead).")

    # Text handling
    p.add_argument("--min_chars", type=int, default=1, help="Drop texts shorter than this")
    p.add_argument("--max_chars", type=int, default=0,
                   help="If >0, truncate text_for_embed to this many chars (reduces CPU time).")

    # Output naming
    p.add_argument("--tag", default="", help="Suffix tag for output filenames (e.g., n500000_seed42)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    return p.parse_args()


def make_out_paths(out_dir: Path, tag: str):
    tag = tag.strip()
    suffix = f"_{tag}" if tag else ""
    emb_path = out_dir / f"embeddings{suffix}.npy"
    ids_path = out_dir / f"ids{suffix}.npy"
    idx_path = out_dir / f"embed_index{suffix}.parquet"
    meta_path = out_dir / f"embed_meta{suffix}.json"
    return emb_path, ids_path, idx_path, meta_path


def _assert_overwrite(paths, overwrite: bool):
    if overwrite:
        return
    exists = [p for p in paths if p.exists()]
    if exists:
        msg = "Output already exists (use --overwrite to replace):\n" + "\n".join([f" - {p}" for p in exists])
        raise FileExistsError(msg)


def _cpu_core_hint() -> int:
    # A reasonable default: physical cores - 1 (fall back to logical)
    try:
        import psutil  # optional
        phys = psutil.cpu_count(logical=False) or 0
        logi = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        if phys >= 2:
            return max(1, phys - 1)
        return max(1, (logi or 1) - 1)
    except Exception:
        c = os.cpu_count() or 1
        return max(1, c - 1)


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path, ids_path, idx_path, meta_path = make_out_paths(out_dir, args.tag)
    _assert_overwrite([emb_path, ids_path, idx_path, meta_path], args.overwrite)

    # Lazy import so errors are clear
    from sentence_transformers import SentenceTransformer

    print(f"[03_embed] Input parquet: {in_path}")
    df = pd.read_parquet(in_path)

    if "id" not in df.columns or "text_for_embed" not in df.columns:
        raise ValueError("Input parquet must contain columns: 'id', 'text_for_embed'")

    # Clean
    df["text_for_embed"] = df["text_for_embed"].fillna("").astype(str)

    if args.max_chars and args.max_chars > 0:
        df["text_for_embed"] = df["text_for_embed"].str.slice(0, args.max_chars)

    df = df[df["text_for_embed"].str.len() >= int(args.min_chars)].copy()
    df.reset_index(drop=True, inplace=True)

    ids = df["id"].astype(np.int64).to_numpy()
    texts = df["text_for_embed"].tolist()
    n = len(texts)

    if n == 0:
        raise ValueError("No texts remaining after filtering (min_chars/max_chars).")

    print(f"[03_embed] Records to embed: {n:,}")

    # Load model (CPU)
    print(f"[03_embed] Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = int(args.max_seq_length)

    # Determine embedding dim with a tiny encode (safe, small)
    print("[03_embed] Probing embedding dimension...")
    probe = model.encode(["probe"], batch_size=1, show_progress_bar=False, convert_to_numpy=True)
    emb_dim = int(probe.shape[1])
    print(f"[03_embed] Embedding dim = {emb_dim}")

    # Prepare memmap output for embeddings
    # Store float32 for downstream stability (UMAP/HDBSCAN)
    print(f"[03_embed] Creating memmap: {emb_path} shape=({n}, {emb_dim}) dtype=float32")
    if emb_path.exists() and args.overwrite:
        emb_path.unlink()
    embs_mm = np.memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, emb_dim))

    # Decide worker count
    if args.device != "cpu":
        print("[03_embed] WARNING: This script is tuned for CPU. You set device != cpu.", file=sys.stderr)

    if args.num_workers and args.num_workers > 0:
        num_workers = int(args.num_workers)
    else:
        num_workers = _cpu_core_hint()

    chunk_size = int(args.chunk_size)
    bs = int(args.batch_size)
    num_chunks = math.ceil(n / chunk_size)

    print(f"[03_embed] Encoding config:")
    print(f"  device={args.device}")
    print(f"  batch_size={bs}")
    print(f"  max_seq_length={args.max_seq_length}")
    print(f"  normalize={bool(args.normalize)}")
    print(f"  num_workers={num_workers}  (0->auto resolved)")
    print(f"  chunk_size={chunk_size}  num_chunks={num_chunks}")

    t_start = time.time()

    # SentenceTransformers CPU speedup option: multi-process pool
    # This can significantly reduce wall time on large n.
    pool = None
    use_pool = (args.device == "cpu") and (num_workers >= 2)

    try:
        if use_pool:
            print("[03_embed] Starting multi-process pool...")
            pool = model.start_multi_process_pool(target_devices=["cpu"] * num_workers)

        write_row = 0
        for c in range(num_chunks):
            s = c * chunk_size
            e = min((c + 1) * chunk_size, n)
            batch_texts = texts[s:e]

            t0 = time.time()
            if pool is not None:
                # Multi-process encode
                chunk_emb = model.encode_multi_process(
                    batch_texts,
                    pool,
                    batch_size=bs,
                )
            else:
                # Single-process encode
                chunk_emb = model.encode(
                    batch_texts,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,  # normalize later if requested
                )

            chunk_emb = np.asarray(chunk_emb, dtype=np.float32)

            # Optional normalize (safe on float32)
            if args.normalize:
                norms = np.linalg.norm(chunk_emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                chunk_emb = chunk_emb / norms

            # Write to memmap
            embs_mm[write_row:write_row + (e - s), :] = chunk_emb
            write_row += (e - s)

            # Flush periodically so partial progress is durable
            if (c % 5) == 0 or c == num_chunks - 1:
                embs_mm.flush()

            dt = time.time() - t0
            rate = (e - s) / max(dt, 1e-6)
            done = e
            elapsed = time.time() - t_start
            print(f"[03_embed] chunk {c+1}/{num_chunks} rows {s:,}-{e-1:,} "
                  f"({e-s:,})  {rate:,.1f} rows/s  elapsed={elapsed/60:.1f} min")

        # Final flush
        embs_mm.flush()

    finally:
        if pool is not None:
            print("[03_embed] Stopping multi-process pool...")
            model.stop_multi_process_pool(pool)

    # Save ids.npy (int64)
    print(f"[03_embed] Saving ids: {ids_path}")
    np.save(ids_path, ids)

    # Save index parquet for traceability
    # id -> row mapping plus run tag/model
    idx_df = pd.DataFrame({
        "row": np.arange(n, dtype=np.int64),
        "id": ids,
    })
    idx_df["model"] = args.model
    idx_df["tag"] = args.tag
    idx_df["max_seq_length"] = int(args.max_seq_length)
    idx_df["normalize"] = bool(args.normalize)
    idx_df["source_parquet"] = str(in_path)

    print(f"[03_embed] Saving embed index parquet: {idx_path}")
    idx_df.to_parquet(idx_path, index=False)

    # Save meta json
    meta = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "n_records": int(n),
        "embedding_dim": int(emb_dim),
        "model": args.model,
        "device": args.device,
        "batch_size": int(bs),
        "chunk_size": int(chunk_size),
        "num_workers": int(num_workers),
        "max_seq_length": int(args.max_seq_length),
        "normalize": bool(args.normalize),
        "max_chars": int(args.max_chars),
        "min_chars": int(args.min_chars),
        "tag": args.tag,
        "stored_dtype": "float32",
        "platform": {
            "python": sys.version,
            "os": platform.platform(),
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t_start),
    }
    print(f"[03_embed] Saving meta: {meta_path}")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[03_embed] DONE. Saved:")
    print(" -", emb_path)
    print(" -", ids_path)
    print(" -", idx_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()