#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap


def parse_args():
    p = argparse.ArgumentParser(description="UMAP 2D projection for embeddings.")
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    p.add_argument("--ids", required=True, help="Path to ids .npy")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--n_neighbors", type=int, default=10)
    p.add_argument("--min_dist", type=float, default=0.0)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings)
    ids = np.load(args.ids)

    print("Embeddings:", emb.shape, emb.dtype)
    print("IDs:", ids.shape)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        verbose=True,
    )

    print(
        f"Running UMAP: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric}, seed={args.seed}"
    )
    xy = reducer.fit_transform(emb)  # (N,2)

    df = pd.DataFrame(
        {
            "id": ids.astype(np.int64),
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
        }
    )

    out_parquet = out_dir / f"umap2d_{args.tag}.parquet"
    df.to_parquet(out_parquet, index=False)

    meta = {
        "tag": args.tag,
        "n": int(df.shape[0]),
        "params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "seed": args.seed,
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