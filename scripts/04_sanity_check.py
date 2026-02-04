#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanity check for sentence embeddings (norms, random cosine stats, approximate top-k cosine)."
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    p.add_argument("--ids", required=True, help="Path to ids .npy")

    # Random pair cosine stats
    p.add_argument("--num_pairs", type=int, default=10000, help="Random pairs to sample for cosine stats")
    p.add_argument("--seed", type=int, default=42)

    # Approximate top-k nearest cosine (helps detect collapse)
    p.add_argument(
        "--num_anchors",
        type=int,
        default=2000,
        help="How many anchor points to sample for approximate top-k cosine stats",
    )
    p.add_argument(
        "--candidates_per_anchor",
        type=int,
        default=2000,
        help="How many random candidates to compare against per anchor (larger = better estimate, slower)",
    )
    p.add_argument("--topk", type=int, default=10, help="Compute top-k cosine among candidates for each anchor")

    p.add_argument("--out", help="Optional path to save stats as json")
    return p.parse_args()


def _safe_l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows to unit norm; safe for zero vectors."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def main():
    args = parse_args()

    emb = np.load(args.embeddings, mmap_mode="r")
    ids = np.load(args.ids, mmap_mode="r")

    print("Embeddings shape:", emb.shape)
    print("Embeddings dtype:", emb.dtype)
    print("IDs shape:", ids.shape, "| dtype:", ids.dtype)

    n, d = emb.shape

    # (C) IDs uniqueness check
    # Note: ids.npy is typically int64; converting to numpy array is cheap relative to embeddings.
    ids_arr = np.asarray(ids)
    unique_ids = np.unique(ids_arr)
    print(f"Unique IDs: {len(unique_ids):,} / {len(ids_arr):,}")
    if len(unique_ids) != len(ids_arr):
        dup_cnt = len(ids_arr) - len(unique_ids)
        print(f"WARNING: Found duplicate IDs: {dup_cnt:,}")

    # Basic numeric checks (NaN/Inf) - note: on memmap, this scans the whole array (O(n*d)).
    # For 100k x 384 this is fine.
    nan_count = int(np.isnan(emb).sum())
    inf_count = int(np.isinf(emb).sum())
    print("NaN count:", nan_count)
    print("Inf count:", inf_count)

    norms = np.linalg.norm(emb, axis=1)
    norm_stats = {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "p1": float(np.percentile(norms, 1)),
        "p50": float(np.percentile(norms, 50)),
        "p99": float(np.percentile(norms, 99)),
    }
    print("L2 norm stats:", norm_stats)

    rng = np.random.default_rng(args.seed)

    # (A) Faster random cosine similarity stats:
    # Vectorized sampling of pairs + normalize only the sampled vectors.
    # This avoids per-pair python loops and repeated norm computations.
    i = rng.integers(0, n, size=args.num_pairs, dtype=np.int64)
    j = rng.integers(0, n, size=args.num_pairs, dtype=np.int64)
    mask = i != j
    i = i[mask]
    j = j[mask]

    # Pull sampled vectors into RAM (small: num_pairs x dim)
    vi = np.asarray(emb[i], dtype=np.float32)
    vj = np.asarray(emb[j], dtype=np.float32)
    vi = _safe_l2_normalize(vi, axis=1)
    vj = _safe_l2_normalize(vj, axis=1)

    sims = np.einsum("nd,nd->n", vi, vj)  # dot row-wise
    cosine_stats = {
        "pairs": int(sims.shape[0]),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "p5": float(np.percentile(sims, 5)),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
    }

    print("\nRandom cosine similarity stats:")
    for k, v in cosine_stats.items():
        if k == "pairs":
            print(f"  {k:>5}: {v}")
        else:
            print(f"  {k:>5}: {v:.4f}")

    # (B) Approximate "top-k nearest cosine" to detect embedding collapse:
    # For each anchor, compare to a random candidate set; compute topk cosines.
    # This is not exact NN, but it is very informative and scales well.
    num_anchors = min(args.num_anchors, n)
    candidates_per_anchor = min(args.candidates_per_anchor, max(1, n - 1))
    topk = max(1, min(args.topk, candidates_per_anchor))

    anchor_idx = rng.choice(n, size=num_anchors, replace=False).astype(np.int64)

    # Collect top1/topk distributions
    top1_list = []
    topk_mean_list = []

    print(
        f"\nApprox top-k cosine (anchors={num_anchors:,}, candidates/anchor={candidates_per_anchor:,}, topk={topk})"
    )

    for a in tqdm(anchor_idx, desc="Anchors"):
        # Sample candidate indices; allow replacement for speed, but avoid self-index.
        cand = rng.integers(0, n, size=candidates_per_anchor, dtype=np.int64)
        cand = cand[cand != a]
        if cand.size == 0:
            continue
        # If we removed too many self matches (rare), resample once.
        if cand.size < candidates_per_anchor:
            extra = rng.integers(0, n, size=(candidates_per_anchor - cand.size), dtype=np.int64)
            extra = extra[extra != a]
            cand = np.concatenate([cand, extra])[:candidates_per_anchor]

        va = np.asarray(emb[a], dtype=np.float32)[None, :]
        vc = np.asarray(emb[cand], dtype=np.float32)

        va = _safe_l2_normalize(va, axis=1)
        vc = _safe_l2_normalize(vc, axis=1)

        cos = (vc @ va.T).ravel()  # (candidates,)

        # partial topk without full sort
        if cos.size <= topk:
            top_vals = np.sort(cos)[::-1]
        else:
            idx = np.argpartition(cos, -topk)[-topk:]
            top_vals = np.sort(cos[idx])[::-1]

        top1_list.append(float(top_vals[0]))
        topk_mean_list.append(float(top_vals.mean()))

    top1 = np.array(top1_list, dtype=np.float32)
    topk_mean = np.array(topk_mean_list, dtype=np.float32)

    approx_nn_stats = {
        "anchors_used": int(top1.size),
        "top1": {
            "min": float(top1.min()) if top1.size else None,
            "max": float(top1.max()) if top1.size else None,
            "mean": float(top1.mean()) if top1.size else None,
            "std": float(top1.std()) if top1.size else None,
            "p5": float(np.percentile(top1, 5)) if top1.size else None,
            "p50": float(np.percentile(top1, 50)) if top1.size else None,
            "p95": float(np.percentile(top1, 95)) if top1.size else None,
        },
        "topk_mean": {
            "min": float(topk_mean.min()) if topk_mean.size else None,
            "max": float(topk_mean.max()) if topk_mean.size else None,
            "mean": float(topk_mean.mean()) if topk_mean.size else None,
            "std": float(topk_mean.std()) if topk_mean.size else None,
            "p5": float(np.percentile(topk_mean, 5)) if topk_mean.size else None,
            "p50": float(np.percentile(topk_mean, 50)) if topk_mean.size else None,
            "p95": float(np.percentile(topk_mean, 95)) if topk_mean.size else None,
        },
    }

    print("\nApprox nearest-neighbor cosine stats (random candidates):")
    print("  top1     :", approx_nn_stats["top1"])
    print("  topk_mean:", approx_nn_stats["topk_mean"])

    stats = {
        "embeddings": {
            "path": str(Path(args.embeddings).resolve()),
            "shape": [int(n), int(d)],
            "dtype": str(emb.dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "l2_norm": norm_stats,
        },
        "ids": {
            "path": str(Path(args.ids).resolve()),
            "count": int(len(ids_arr)),
            "unique_count": int(len(unique_ids)),
        },
        "random_cosine": cosine_stats,
        "approx_topk_cosine": {
            "num_anchors": int(num_anchors),
            "candidates_per_anchor": int(candidates_per_anchor),
            "topk": int(topk),
            **approx_nn_stats,
        },
        "seed": int(args.seed),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("\nSaved stats to:", out_path)


if __name__ == "__main__":
    main()