#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap

from sklearn.decomposition import IncrementalPCA, PCA


def parse_args():
    p = argparse.ArgumentParser(description="UMAP 2D projection for large-scale embeddings.")
    # I/O
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    p.add_argument("--ids", required=True, help="Path to ids .npy")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", type=str, default="run")

    # UMAP params
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    # Optimization
    p.add_argument("--pca_dim", type=int, default=50, help="PCA dims before UMAP. 0=skip PCA.")
    p.add_argument("--pca_mode", choices=["incremental", "full"], default="incremental",
                   help="incremental=streaming PCA (recommended for large). full=PCA.fit_transform (may need lots of RAM).")
    p.add_argument("--pca_batch", type=int, default=50000, help="Batch size for IncrementalPCA partial_fit/transform.")

    p.add_argument("--n_jobs", type=int, default=-1, help="UMAP n_jobs. -1=all cores")

    # low_memory
    p.add_argument("--low_memory", action="store_true", help="Force low_memory=True.")
    p.add_argument("--no_low_memory", action="store_true", help="Set low_memory=False (may be faster, more RAM).")

    # Large-scale strategy
    p.add_argument("--subsample_fit", type=int, default=None,
                   help="Fit UMAP on N subsampled points, then transform all.")
    p.add_argument("--batch_transform", type=int, default=100000,
                   help="Batch size for transform when using subsample_fit.")

    # Memory control
    p.add_argument("--force_float32", action="store_true",
                   help="If set, embeddings will be copied to float32 (RAM heavy). Default keeps memmap dtype as-is.")
    return p.parse_args()


def _make_reducer(args, low_memory: bool):
    # UMAP transform uses learned graph; keep random_state fixed.
    return umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        low_memory=low_memory,
        verbose=True,
    )


def apply_pca_incremental(emb_mm, pca_dim: int, seed: int, batch: int):
    """
    Streaming PCA using IncrementalPCA:
    - partial_fit over batches (does not require loading full emb in RAM)
    - then transform over batches to build reduced array (float32)
    """
    n_total, d = emb_mm.shape
    print(f"\n[PCA:incremental] {d}D → {pca_dim}D  n={n_total:,}  batch={batch:,}")
    t0 = time.time()

    ipca = IncrementalPCA(n_components=pca_dim)

    # pass 1: partial_fit
    for i in range(0, n_total, batch):
        j = min(i + batch, n_total)
        X = np.asarray(emb_mm[i:j], dtype=np.float32)  # small batch to float32
        ipca.partial_fit(X)
        if (i // batch) % 5 == 0:
            print(f"  partial_fit {j:,}/{n_total:,}")

    t_fit = time.time() - t0
    print(f"[PCA:incremental] partial_fit done in {t_fit/60:.1f} min")

    # pass 2: transform to a dense float32 array (this is unavoidable if we want UMAP on reduced vectors)
    out = np.empty((n_total, pca_dim), dtype=np.float32)
    t1 = time.time()
    for i in range(0, n_total, batch):
        j = min(i + batch, n_total)
        X = np.asarray(emb_mm[i:j], dtype=np.float32)
        out[i:j] = ipca.transform(X).astype(np.float32)
        if (i // batch) % 5 == 0:
            print(f"  transform {j:,}/{n_total:,}")

    t_tr = time.time() - t1
    print(f"[PCA:incremental] transform done in {t_tr/60:.1f} min")
    print(f"[PCA:incremental] total { (time.time()-t0)/60:.1f} min")
    return out


def apply_pca_full(emb, pca_dim: int, seed: int):
    """
    Full PCA (RAM heavy). Use only if you have ample memory.
    """
    print(f"\n[PCA:full] {emb.shape[1]}D → {pca_dim}D")
    start = time.time()
    pca = PCA(n_components=pca_dim, random_state=seed)
    out = pca.fit_transform(np.asarray(emb, dtype=np.float32)).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum() * 100.0)
    elapsed = time.time() - start
    print(f"[PCA:full] Explained variance: {explained:.1f}%")
    print(f"[PCA:full] Done in {elapsed/60:.1f} min")
    return out


def fit_transform_full(emb2d_input, args, low_memory: bool):
    print(f"\n[UMAP] Full fit_transform on {len(emb2d_input):,} points")
    reducer = _make_reducer(args, low_memory)
    start = time.time()
    xy = reducer.fit_transform(emb2d_input).astype(np.float32)
    elapsed = time.time() - start
    print(f"[UMAP] Done in {elapsed/60:.1f} min")
    return xy


def fit_transform_subsampled(emb2d_input, args, low_memory: bool):
    n_total = len(emb2d_input)
    n_fit = int(args.subsample_fit)

    print(f"\n[UMAP] Subsample strategy: fit on {n_fit:,}, transform {n_total:,}")

    rng = np.random.default_rng(args.seed)
    fit_indices = rng.choice(n_total, size=n_fit, replace=False)
    fit_indices.sort()

    emb_fit = emb2d_input[fit_indices]
    reducer = _make_reducer(args, low_memory)

    print(f"\n[UMAP] Fitting on {n_fit:,} subsampled points...")
    start_fit = time.time()
    reducer.fit(emb_fit)
    fit_time = time.time() - start_fit
    print(f"[UMAP] Fit done in {fit_time/60:.1f} min")

    print(f"\n[UMAP] Transforming all {n_total:,} points in batches...")
    start_transform = time.time()

    batch_size = int(args.batch_transform)
    xy = np.empty((n_total, 2), dtype=np.float32)

    for i in range(0, n_total, batch_size):
        end = min(i + batch_size, n_total)
        xy[i:end] = reducer.transform(emb2d_input[i:end]).astype(np.float32)
        progress = end / n_total * 100.0
        print(f"  Transformed {end:,}/{n_total:,} ({progress:.1f}%)")

    transform_time = time.time() - start_transform
    print(f"[UMAP] Transform done in {transform_time/60:.1f} min")
    return xy


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    low_memory = True
    if args.no_low_memory:
        low_memory = False
    if args.low_memory:
        low_memory = True

    meta_path = Path(args.embeddings).parent / Path(args.embeddings).name.replace("embeddings_", "embed_meta_").replace(".npy", ".json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    n_records = meta["n_records"]
    emb_dim = meta["embedding_dim"]

    emb_mm = np.memmap(args.embeddings, dtype='float32', mode='r', shape=(n_records, emb_dim))
    ids_mm = np.load(args.ids, mmap_mode="r")  # .npy header-aware
    if ids_mm.dtype != np.int64:
        ids_mm = ids_mm.astype(np.int64, copy=False)
    if ids_mm.shape[0] != n_records:
        raise ValueError(f"ids length mismatch: ids={ids_mm.shape[0]} vs n_records={n_records}")

    print("Embeddings:", emb_mm.shape, emb_mm.dtype)
    print("IDs:", ids_mm.shape, ids_mm.dtype)

    n_total = int(len(emb_mm))
    total_start = time.time()

    # Keep memmap by default (do NOT np.asarray the entire thing)
    if args.force_float32:
        print("[WARN] --force_float32 enabled: copying full embeddings into RAM as float32.")
        emb_work = np.asarray(emb_mm, dtype=np.float32)
    else:
        emb_work = emb_mm  # memmap

    # Step 1: PCA (optional)
    if args.pca_dim and args.pca_dim > 0:
        pca_dim = int(args.pca_dim)
        if args.pca_mode == "incremental":
            emb_reduced = apply_pca_incremental(emb_work, pca_dim=pca_dim, seed=int(args.seed), batch=int(args.pca_batch))
        else:
            emb_reduced = apply_pca_full(emb_work, pca_dim=pca_dim, seed=int(args.seed))
        emb_for_umap = emb_reduced
    else:
        # If skipping PCA: ensure float32 view per batch is okay.
        # UMAP will likely touch lots of memory; consider PCA for stability/perf.
        print("\n[INFO] PCA skipped. UMAP will run on original embedding dimension.")
        emb_for_umap = np.asarray(emb_work, dtype=np.float32) if isinstance(emb_work, np.memmap) else emb_work

    # Step 2: UMAP
    if args.subsample_fit and int(args.subsample_fit) < n_total:
        xy = fit_transform_subsampled(emb_for_umap, args, low_memory)
    else:
        xy = fit_transform_full(emb_for_umap, args, low_memory)

    total_time = time.time() - total_start
    print(f"\n[TOTAL] Completed in {total_time/60:.1f} min")

    n_unique = np.unique(np.asarray(ids_mm)).size
    if n_unique != n_total:
        raise ValueError(f"ID is not unique: duplicates={n_total - n_unique}")

    # Save outputs
    df = pd.DataFrame({
        "id": np.asarray(ids_mm, dtype=np.int64),
        "x": xy[:, 0].astype(np.float32),
        "y": xy[:, 1].astype(np.float32),
    })

    out_parquet = out_dir / f"umap2d_{args.tag}.parquet"
    df.to_parquet(out_parquet, index=False)

    meta = {
        "tag": args.tag,
        "n": int(n_total),
        "total_time_min": round(total_time / 60, 2),
        "params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "seed": args.seed,
            "pca_dim": args.pca_dim,
            "pca_mode": args.pca_mode,
            "pca_batch": args.pca_batch,
            "n_jobs": args.n_jobs,
            "low_memory": low_memory,
            "subsample_fit": args.subsample_fit,
            "batch_transform": args.batch_transform,
            "force_float32": bool(args.force_float32),
        },
        "inputs": {"embeddings": str(Path(args.embeddings)), "ids": str(Path(args.ids))},
        "outputs": {"umap2d_parquet": str(out_parquet)},
    }

    out_meta = out_dir / f"umap2d_{args.tag}.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", out_parquet)
    print(" -", out_meta)


if __name__ == "__main__":
    main()