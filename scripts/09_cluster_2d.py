#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Cluster in 2D UMAP space (HDBSCAN)")

    p.add_argument("--umap", required=True, help="UMAP parquet with columns: id, x, y")
    p.add_argument("--texts", required=True, help="Texts parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory (e.g., web/public/packed)")
    p.add_argument("--min_cluster_size", type=int, default=80)
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--tag", required=True)

    # 분리/조각화 성향 옵션
    p.add_argument("--leaf", action="store_true", help="Use cluster_selection_method=leaf (finer clusters)")
    p.add_argument("--allow_single_cluster", action="store_true", help="Allow single-cluster solution (default: False)")

    return p.parse_args()


def remap_to_uint16(labels_raw: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    HDBSCAN labels: -1 noise, 0..K-1 clusters.
    Output uint16: clusters keep 0..K-1, noise becomes K (noise_bucket).
    Returns: (cluster_u16, num_clusters, noise_count)
    """
    labels_raw = labels_raw.astype(np.int64, copy=False)
    uniq = np.unique(labels_raw)

    clusters = [u for u in uniq.tolist() if u != -1]
    clusters.sort()
    num_clusters = len(clusters)
    noise_bucket = num_clusters

    out = np.empty(labels_raw.shape[0], dtype=np.uint16)

    if num_clusters == 0:
        # 전부 노이즈면 0으로 통일(=noise_bucket도 0)
        out[:] = 0
        return out, 0, int((labels_raw == -1).sum())

    mapping = {c: i for i, c in enumerate(clusters)}
    noise = 0
    for i, v in enumerate(labels_raw.tolist()):
        if v == -1:
            out[i] = noise_bucket
            noise += 1
        else:
            out[i] = mapping.get(v, noise_bucket)
    return out, num_clusters, noise


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    umap_path = Path(args.umap)
    texts_path = Path(args.texts)

    # --- Load UMAP ---
    dfu = pd.read_parquet(umap_path)
    need_cols = {"id", "x", "y"}
    if not need_cols.issubset(dfu.columns):
        raise ValueError(f"--umap parquet must contain columns {sorted(need_cols)}; got {dfu.columns.tolist()}")

    dfu = dfu[["id", "x", "y"]].copy()
    dfu["id"] = dfu["id"].astype(np.int64)

    # --- Load texts (for keyword extraction / labels) ---
    dft = pd.read_parquet(texts_path)
    if "id" not in dft.columns:
        raise ValueError("--texts parquet must contain column: id")
    if "text_for_embed" not in dft.columns:
        # 라벨 생성 로직이 text_for_embed를 쓴다고 가정
        raise ValueError("--texts parquet must contain column: text_for_embed")

    dft = dft[["id", "text_for_embed"]].copy()
    dft["id"] = dft["id"].astype(np.int64)

    # UMAP id 기준으로 정렬/정합
    df = dfu.merge(dft, on="id", how="left", validate="one_to_one")
    if df.shape[0] == 0:
        raise ValueError("Joined dataframe is empty. Check id keys between --umap and --texts.")
    df["text_for_embed"] = df["text_for_embed"].fillna("").astype(str)

    n = df.shape[0]
    print(f"Loaded {n} points from UMAP")

    # --- Build X ALWAYS (this fixes your error) ---
    X = df[["x", "y"]].to_numpy(dtype=np.float32)

    # --- Cluster ---
    import hdbscan

    cluster_selection_method = "leaf" if args.leaf else "eom"
    allow_single_cluster = bool(args.allow_single_cluster)  # default False

    print(
        f"Running HDBSCAN: min_cluster_size={args.min_cluster_size}, "
        f"min_samples={args.min_samples}, method={cluster_selection_method}, "
        f"allow_single_cluster={allow_single_cluster}"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )

    labels_raw = clusterer.fit_predict(X)

    cluster_u16, num_clusters, noise = remap_to_uint16(labels_raw)
    noise_pct = (noise / n) * 100.0 if n else 0.0
    print(f"Clusters: {num_clusters}, Noise: {noise} ({noise_pct:.1f}%)")

    # --- Save cluster buffer ---
    cluster_path = out_dir / f"cluster_{args.tag}.uint16"
    cluster_u16.tofile(cluster_path)
    print("Saved:", cluster_path)

    # --- Save labels json (simple + consistent schema) ---
    # 여기서는 "labels"를 비워두고, 기존 11_make_cluster_labels.py가 있다면 그걸로 키워드 라벨을 만들게 하는 방식이 안정적.
    # 하지만 현재 웹이 labelsJson.num_clusters / noise_bucket / labels 를 기대하므로 최소 형태는 제공.
    labels_json = {
        "tag": args.tag,
        "n": int(n),
        "num_clusters": int(num_clusters),
        "noise_bucket": int(num_clusters if num_clusters > 0 else 0),
        "labels": {},  # 키워드 라벨은 11_make_cluster_labels.py로 채우는 것을 권장
    }

    labels_path = out_dir / f"cluster_labels_{args.tag}.json"
    labels_path.write_text(json.dumps(labels_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", labels_path)


if __name__ == "__main__":
    main()