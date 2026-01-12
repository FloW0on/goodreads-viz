#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate cluster_labels_<tag>.json from:
- ids (npy) defining point order
- cluster assignment (uint16 binary) OR raw labels (npy)
- texts parquet (id, text_for_embed)

Output JSON schema is compatible with current webgpu_points.js:
{
  "n": N,
  "method": "...",
  "params": {...},
  "num_clusters": C,
  "noise_bucket": C,
  "labels": {
     "0": {"size":..., "keywords":[...], "label":"..."},
     ...
     "C": {"size": noise_count, "keywords":["noise"], "label":"noise/outliers"}
  }
}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser(description="Make TF-IDF cluster label JSON for WebGPU legend.")
    p.add_argument("--ids_npy", required=True, help="ids_n*.npy (point order used by points_xy/ids buffers)")
    p.add_argument("--texts", required=True, help="Texts parquet with columns: id, text_for_embed")
    p.add_argument("--out_dir", required=True, help="Output directory (packed)")
    p.add_argument("--tag", required=True, help="Suffix tag for output filenames")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cluster_uint16", help="cluster_<tag>.uint16 file (already remapped, noise==C)")
    g.add_argument("--raw_labels_npy", help="raw labels npy (e.g., hdbscan labels in [-1..]) to be remapped")

    # TF-IDF options
    p.add_argument("--max_features", type=int, default=10000)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=3)
    p.add_argument("--stop_words", type=str, default="english")

    p.add_argument("--topk", type=int, default=8, help="Top K keywords per cluster")
    p.add_argument("--label_k", type=int, default=4, help="How many keywords to join as a short label")
    p.add_argument("--noise_label", type=str, default="noise/outliers")

    # bookkeeping for metadata
    p.add_argument("--method", type=str, default="tfidf_cluster_labels")
    p.add_argument("--params_json", type=str, default="", help="Optional JSON string to include under 'params'")
    return p.parse_args()


def _load_cluster_from_uint16(path: Path, n: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    if arr.shape[0] != n:
        raise ValueError(f"cluster_uint16 length mismatch: {arr.shape[0]} vs n={n}")
    return arr


def _remap_raw_labels(raw: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    raw labels: -1 for noise, others arbitrary ints
    returns:
      cluster_out uint16 with noise mapped to C
      C (#clusters excluding noise)
      noise_count
    """
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    raw = raw.astype(np.int64, copy=False)

    unique_labels = sorted([c for c in np.unique(raw) if c >= 0])
    C = len(unique_labels)
    remap = {c: i for i, c in enumerate(unique_labels)}

    out = np.empty(raw.shape[0], dtype=np.uint16)
    noise_count = int((raw < 0).sum())
    for i, c in enumerate(raw):
        out[i] = C if c < 0 else remap[int(c)]
    return out, C, noise_count


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.load(args.ids_npy)
    if ids.ndim != 1:
        ids = ids.reshape(-1)
    n = int(ids.shape[0])

    # 1) cluster assignment in the same point order as ids
    if args.cluster_uint16:
        cluster_path = Path(args.cluster_uint16)
        cluster = _load_cluster_from_uint16(cluster_path, n)
        # infer C and noise from content:
        # assume noise is the max id (C) as in your pipeline
        max_id = int(cluster.max())
        # If noise exists, it should be == max_id, but noise may be 0 if none.
        # We'll treat "noise bucket" as max_id if it appears AND is not a regular cluster
        counts = np.bincount(cluster.astype(np.int64))
        noise_bucket = max_id
        noise_count = int(counts[noise_bucket]) if noise_bucket < len(counts) else 0
        C = noise_bucket  # clusters are 0..C-1, noise==C
    else:
        raw = np.load(args.raw_labels_npy)
        cluster, C, noise_count = _remap_raw_labels(raw)
        noise_bucket = C

    # 2) load texts and align by ids order
    texts_df = pd.read_parquet(args.texts)
    if "id" not in texts_df.columns or "text_for_embed" not in texts_df.columns:
        raise ValueError("texts parquet must have columns: id, text_for_embed")

    texts_df = texts_df.drop_duplicates("id").set_index("id")
    # Ensure all ids exist in texts_df
    missing = [int(i) for i in ids if int(i) not in texts_df.index]
    if missing:
        # show only a few to avoid huge logs
        ex = missing[:10]
        raise KeyError(f"Missing {len(missing)} ids in texts parquet. Examples: {ex}")

    texts = texts_df.loc[ids, "text_for_embed"].astype(str).tolist()

    # 3) TF-IDF
    vec = TfidfVectorizer(
        max_features=args.max_features,
        stop_words=args.stop_words if args.stop_words else None,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    # 4) keywords per cluster
    labels_out: dict[str, dict] = {}

    # regular clusters 0..C-1
    for k in range(C):
        idx = np.where(cluster == k)[0]
        if idx.size == 0:
            labels_out[str(k)] = {"size": 0, "keywords": [], "label": ""}
            continue
        m = np.asarray(X[idx].mean(axis=0)).ravel()
        top = np.argsort(-m)[: args.topk]
        kws = vocab[top].tolist()
        labels_out[str(k)] = {
            "size": int(idx.size),
            "keywords": kws,
            "label": " / ".join(kws[: args.label_k]),
        }

    # noise bucket
    labels_out[str(noise_bucket)] = {
        "size": int(noise_count),
        "keywords": ["noise"],
        "label": args.noise_label,
    }

    # 5) assemble JSON
    params = {}
    if args.params_json:
        try:
            params = json.loads(args.params_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"--params_json is not valid JSON: {e}") from e

    payload = {
        "n": n,
        "method": args.method,
        "params": params,
        "num_clusters": int(C),
        "noise_bucket": int(noise_bucket),
        "labels": labels_out,
    }

    out_path = out_dir / f"cluster_labels_{args.tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()