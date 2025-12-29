#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Unsupervised topic clustering (HDBSCAN/KMeans) + TF-IDF keyword labels."
    )
    p.add_argument("--embeddings", required=True, help="Embeddings .npy (NxD)")
    p.add_argument("--ids", required=True, help="IDs .npy (N,) int/uint")
    p.add_argument("--texts", required=True, help="Parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory (recommend: processed/packed)")
    p.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    p.add_argument("--k", type=int, default=40, help="K for KMeans (ignored for hdbscan)")
    p.add_argument("--min_cluster_size", type=int, default=40, help="HDBSCAN min_cluster_size")
    p.add_argument("--min_samples", type=int, default=10, help="HDBSCAN min_samples")
    p.add_argument("--max_features", type=int, default=20000, help="TF-IDF max features")
    p.add_argument("--topk", type=int, default=8, help="Top keywords per cluster")
    p.add_argument("--tag", default="n10000_seed42")
    return p.parse_args()


def build_tfidf_labels(texts: list[str], labels_raw: np.ndarray, topk: int, max_features: int):
    """
    Build keyword labels per raw cluster id using TF-IDF.
    labels_raw: original clustering labels (noise = -1)
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    out = {}
    uniq = np.unique(labels_raw)
    for c in uniq:
        if c < 0:
            continue
        idx = np.where(labels_raw == c)[0]
        if len(idx) == 0:
            continue
        m = X[idx].mean(axis=0)
        m = np.asarray(m).ravel()
        top = np.argsort(-m)[:topk]
        kws = vocab[top].tolist()
        out[int(c)] = {
            "size": int(len(idx)),
            "keywords": kws,
            "label": " / ".join(kws[:4]),
        }
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings, mmap_mode="r")  
    ids = np.load(args.ids)  
    ids = ids.astype(np.int64, copy=False)

    df = pd.read_parquet(args.texts)
    if "id" not in df.columns or "text_for_embed" not in df.columns:
        raise ValueError("texts parquet must have columns: id, text_for_embed")

    df = df[["id", "text_for_embed"]].copy()
    df["id"] = df["id"].astype(np.int64)
    df = df.set_index("id")

    missing = [int(x) for x in ids if int(x) not in df.index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} ids in texts parquet (example: {missing[:5]})")

    texts = df.loc[ids, "text_for_embed"].astype(str).tolist()

    if args.method == "hdbscan":
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
        )
        labels_raw = clusterer.fit_predict(np.asarray(emb))
    else:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(
            n_clusters=args.k, batch_size=4096, random_state=42, n_init="auto"
        )
        labels_raw = km.fit_predict(np.asarray(emb))

    labels_raw = labels_raw.astype(np.int32)

    uniq = sorted([int(c) for c in np.unique(labels_raw) if c >= 0])
    remap = {c: i for i, c in enumerate(uniq)}
    C = len(uniq)
    noise = int((labels_raw < 0).sum())

    cluster = np.empty(labels_raw.shape[0], dtype=np.uint16)
    for i, c in enumerate(labels_raw):
        cluster[i] = C if c < 0 else remap[int(c)]

    label_info_raw = build_tfidf_labels(texts, labels_raw, args.topk, args.max_features)

    labels_out = {}
    for raw_c, info in label_info_raw.items():
        labels_out[str(remap[int(raw_c)])] = info
    labels_out[str(C)] = {"size": noise, "keywords": ["noise"], "label": "noise/outliers"}

    cluster_path = out_dir / f"cluster_{args.tag}.uint16"
    labels_path = out_dir / f"cluster_labels_{args.tag}.json"

    cluster.tofile(cluster_path)
    labels_path.write_text(
        json.dumps(
            {
                "n": int(cluster.shape[0]),
                "method": args.method,
                "num_clusters": int(C),
                "noise_bucket": int(C),
                "labels": labels_out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:")
    print(" -", cluster_path)
    print(" -", labels_path)
    print(f"clusters={C}, noise={noise}, total={cluster.shape[0]}")


if __name__ == "__main__":
    main()