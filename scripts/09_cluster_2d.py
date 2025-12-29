#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cluster points in 2D UMAP space using HDBSCAN."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser(description="Cluster in 2D UMAP space")
    p.add_argument("--umap", required=True, help="UMAP parquet (id, x, y)")
    p.add_argument("--texts", required=True, help="Texts parquet (id, text_for_embed)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--min_cluster_size", type=int, default=100)
    p.add_argument("--min_samples", type=int, default=20)
    p.add_argument("--tag", type=str, default="run")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load UMAP 2D coordinates
    umap_df = pd.read_parquet(args.umap)
    xy = umap_df[["x", "y"]].values
    ids = umap_df["id"].values
    print(f"Loaded {len(xy)} points from UMAP")

    # 2. HDBSCAN on 2D
    print(f"Running HDBSCAN: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(xy)

    unique_labels = sorted([c for c in np.unique(labels) if c >= 0])
    num_clusters = len(unique_labels)
    noise_count = int((labels == -1).sum())
    print(f"Clusters: {num_clusters}, Noise: {noise_count} ({100*noise_count/len(labels):.1f}%)")

    # 3. Load texts for keyword extraction
    texts_df = pd.read_parquet(args.texts).drop_duplicates("id").set_index("id")
    texts = texts_df.loc[ids, "text_for_embed"].astype(str).tolist()

    # 4. TF-IDF keyword extraction
    vec = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    # 5. Remap cluster IDs (noise -> last bucket)
    remap = {c: i for i, c in enumerate(unique_labels)}
    C = num_clusters

    cluster_out = np.empty(len(labels), dtype=np.uint16)
    for i, c in enumerate(labels):
        cluster_out[i] = C if c < 0 else remap[int(c)]

    # 6. Extract keywords per cluster
    labels_out = {}
    for raw_c in unique_labels:
        idx = np.where(labels == raw_c)[0]
        m = np.asarray(X[idx].mean(axis=0)).ravel()
        top = np.argsort(-m)[:8]
        kws = vocab[top].tolist()
        labels_out[str(remap[raw_c])] = {
            "size": int(len(idx)),
            "keywords": kws,
            "label": " / ".join(kws[:4]),
        }

    labels_out[str(C)] = {"size": noise_count, "keywords": ["noise"], "label": "noise/outliers"}

    # 7. Save outputs
    cluster_path = out_dir / f"cluster_{args.tag}.uint16"
    labels_path = out_dir / f"cluster_labels_{args.tag}.json"

    cluster_out.tofile(cluster_path)
    print(f"Saved: {cluster_path}")

    with open(labels_path, "w") as f:
        json.dump(
            {
                "n": len(cluster_out),
                "method": "hdbscan_2d",
                "params": {"min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples},
                "num_clusters": C,
                "noise_bucket": C,
                "labels": labels_out,
            },
            f,
            indent=2,
        )
    print(f"Saved: {labels_path}")


if __name__ == "__main__":
    main()