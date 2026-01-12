#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap


def parse_args():
    p = argparse.ArgumentParser("Supervised UMAP using cluster labels")
    p.add_argument("--embeddings", required=True, help="embeddings .npy (N,D)")
    p.add_argument("--ids", required=True, help="ids .npy (N,)")
    p.add_argument("--cluster_uint16", required=True, help="cluster labels .uint16")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--target_weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings).astype(np.float32)
    ids = np.load(args.ids).astype(np.int64)
    cluster = np.fromfile(args.cluster_uint16, dtype=np.uint16)

    assert len(emb) == len(ids) == len(cluster)

    print("Embeddings:", emb.shape)
    print("Clusters:", np.unique(cluster).size, "unique")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        target_metric="categorical",
        target_weight=args.target_weight,
        random_state=args.seed,
        verbose=True,
    )

    print(
        f"Running Supervised UMAP: "
        f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, "
        f"target_weight={args.target_weight}"
    )

    xy = reducer.fit_transform(emb, y=cluster)

    df = pd.DataFrame(
        {
            "id": ids,
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
        }
    )

    out_parquet = out / f"umap2d_supervised_{args.tag}.parquet"
    df.to_parquet(out_parquet, index=False)

    meta = {
        "tag": args.tag,
        "n": int(len(df)),
        "method": "umap_supervised",
        "params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "target_weight": args.target_weight,
            "metric": "cosine",
            "target_metric": "categorical",
            "seed": args.seed,
        },
        "inputs": {
            "embeddings": str(Path(args.embeddings)),
            "cluster": str(Path(args.cluster_uint16)),
        },
        "outputs": {"umap2d_parquet": str(out_parquet)},
    }

    meta_path = out / f"umap2d_supervised_{args.tag}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", out_parquet)
    print(" -", meta_path)


if __name__ == "__main__":
    main()