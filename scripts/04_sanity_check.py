#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanity check for sentence embeddings (cosine similarity stats)."
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    p.add_argument("--ids", required=True, help="Path to ids .npy")
    p.add_argument("--num_pairs", type=int, default=10000, help="Random pairs to sample")
    p.add_argument("--out", help="Optional path to save stats as json")
    return p.parse_args()


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    args = parse_args()

    emb = np.load(args.embeddings, mmap_mode="r")
    ids = np.load(args.ids, mmap_mode="r")

    print("Embeddings shape:", emb.shape)
    print("Embeddings dtype:", emb.dtype)
    print("IDs shape:", ids.shape)

    # Basic checks
    print("NaN count:", np.isnan(emb).sum())
    print("Inf count:", np.isinf(emb).sum())

    norms = np.linalg.norm(emb, axis=1)
    print(
        "L2 norm stats:",
        {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
            "std": float(norms.std()),
        },
    )

    # Random cosine similarities
    n = emb.shape[0]
    rng = np.random.default_rng(42)

    sims = []
    for _ in tqdm(range(args.num_pairs), desc="Sampling cosine similarities"):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(cosine_sim(emb[i], emb[j]))

    sims = np.array(sims)

    stats = {
        "pairs": len(sims),
        "cosine_sim": {
            "min": float(sims.min()),
            "max": float(sims.max()),
            "mean": float(sims.mean()),
            "std": float(sims.std()),
            "p5": float(np.percentile(sims, 5)),
            "p50": float(np.percentile(sims, 50)),
            "p95": float(np.percentile(sims, 95)),
        },
    }

    print("\nCosine similarity stats:")
    for k, v in stats["cosine_sim"].items():
        print(f"  {k:>5}: {v:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("\nSaved stats to:", out_path)


if __name__ == "__main__":
    main()