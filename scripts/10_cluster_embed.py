#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 고차원 임베딩에서 클러스터링
# 실제 의미적 유사도 기반

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser("HDBSCAN on embedding space -> uint16 cluster buffer")

    # 기존 parquet 입력(호환 유지)
    p.add_argument(
        "--embed",
        default=None,
        help="Embedding parquet (id, embedding). If provided, ignores --embeddings_npy/--ids_npy.",
    )

    # 신규 npy 입력
    p.add_argument("--embeddings_npy", default=None, help="embeddings_*.npy (N,D)")
    p.add_argument("--ids_npy", default=None, help="ids_*.npy (N,)")

    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_cluster_size", type=int, default=20)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--normalize", action="store_true", help="L2 normalize embeddings before clustering (recommended for cosine-like behavior).")
    p.add_argument("--tag", required=True)

    return p.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def load_from_parquet(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(path)
    if "id" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Parquet must have columns: id, embedding")

    ids = df["id"].to_numpy(dtype=np.int64)

    # embedding 컬럼이 list/np.ndarray 형태로 들어있다고 가정
    emb0 = df["embedding"].iloc[0]
    if isinstance(emb0, (list, tuple, np.ndarray)):
        embs = np.vstack(df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32)).to_list())
    else:
        raise ValueError("Parquet column 'embedding' must be list/array per row")

    return ids, embs.astype(np.float32)


def load_from_npy(embeddings_npy: Path, ids_npy: Path) -> Tuple[np.ndarray, np.ndarray]:
    embs = np.load(embeddings_npy).astype(np.float32)
    ids = np.load(ids_npy)

    if embs.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N,D). got {embs.shape}")
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D (N,). got {ids.shape}")
    if embs.shape[0] != ids.shape[0]:
        raise ValueError(f"length mismatch: embs={embs.shape[0]} ids={ids.shape[0]}")

    return ids.astype(np.int64), embs


def remap_to_uint16(labels_raw: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    HDBSCAN labels: -1 is noise, 0..K-1 are clusters.
    We remap to uint16 where:
      - clusters keep 0..K-1
      - noise becomes K (noise_bucket)
    Returns: (cluster_uint16, num_clusters)
    """
    uniq = np.unique(labels_raw)
    clusters = [u for u in uniq.tolist() if u != -1]
    clusters = sorted(clusters)
    num_clusters = len(clusters)
    noise_bucket = num_clusters

    out = np.empty(labels_raw.shape[0], dtype=np.uint16)
    if num_clusters == 0:
        out[:] = 0
        return out, 0

    # labels_raw already 0..K-1 in normal HDBSCAN output, but be robust:
    mapping = {c: i for i, c in enumerate(clusters)}
    for i, v in enumerate(labels_raw.tolist()):
        if v == -1:
            out[i] = noise_bucket
        else:
            out[i] = mapping.get(v, noise_bucket)
    return out, num_clusters


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load embeddings ---
    if args.embed:
        ids, emb = load_from_parquet(Path(args.embed))
        src_desc = f"parquet:{args.embed}"
    else:
        if not args.embeddings_npy or not args.ids_npy:
            raise SystemExit(
                "Provide either --embed (parquet) OR both --embeddings_npy and --ids_npy."
            )
        ids, emb = load_from_npy(Path(args.embeddings_npy), Path(args.ids_npy))
        src_desc = f"npy:{args.embeddings_npy} + {args.ids_npy}"

    n, d = emb.shape
    print(f"Loaded embeddings: n={n:,} d={d} from {src_desc}")

    # optional normalize
    want_cosine = (args.metric == "cosine")
    if args.normalize or want_cosine:
        emb = l2_normalize(emb)
        print("Applied L2 normalization")

    # --- Cluster ---
    import hdbscan

    # BallTree는 cosine을 못 받으므로:
    # - cosine 요청이면 L2 정규화 + euclidean로 실행
    hdb_metric = "euclidean" if want_cosine else args.metric

    print(
        f"Running HDBSCAN: min_cluster_size={args.min_cluster_size}, "
        f"min_samples={args.min_samples}, metric={args.metric} (effective={hdb_metric})"
    )

    labels_raw = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=hdb_metric,
    ).fit_predict(emb)

    # remap to uint16 (noise -> K)
    cluster_u16, num_clusters = remap_to_uint16(labels_raw)
    noise = int((labels_raw == -1).sum())
    print(f"Clusters: {num_clusters} | noise: {noise} ({100.0*noise/n:.1f}%)")

    # --- Save ---
    cluster_path = out_dir / f"cluster_{args.tag}.uint16"
    cluster_u16.tofile(cluster_path)
    print("Saved:", cluster_path)

    meta = {
        "tag": args.tag,
        "n": int(n),
        "embedding_dim": int(d),
        "source": src_desc,
        "hdbscan": {
            "min_cluster_size": int(args.min_cluster_size),
            "min_samples": int(args.min_samples),
            "metric": args.metric,
            "effective_metric": hdb_metric,
            "normalize": bool(args.normalize or want_cosine),
            "num_clusters": int(num_clusters),
            "noise_bucket": int(num_clusters),
            "noise_count": int(noise),
        },
    }
    meta_path = out_dir / f"cluster_meta_{args.tag}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", meta_path)



if __name__ == "__main__":
    main()