#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_cluster_2d.py
UMAP 2D 좌표 기반 HDBSCAN 클러스터링 (노이즈 할당 옵션 포함)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import hdbscan


def parse_args():
    p = argparse.ArgumentParser(description="Cluster 2D UMAP points with HDBSCAN")
    p.add_argument("--umap", required=True, help="UMAP parquet (columns: id, x, y)")
    p.add_argument("--texts", required=True, help="Texts parquet for keyword extraction")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", required=True, help="Output tag")
    
    # HDBSCAN 파라미터
    p.add_argument("--min_cluster_size", type=int, default=15)
    p.add_argument("--min_samples", type=int, default=3)
    p.add_argument("--cluster_selection_method", default="eom", choices=["eom", "leaf"])
    p.add_argument("--allow_single_cluster", action="store_true")
    
    # 노이즈 할당 옵션
    p.add_argument("--assign_noise", action="store_true",
                   help="Assign noise points to nearest cluster")
    #packed_ids 정렬 저장
    p.add_argument("--packed_ids_npy", default=None,
               help="Optional packed ids .npy; if provided, output is aligned to this id order.")
    
    return p.parse_args()


def assign_noise_to_nearest(xy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    노이즈 포인트(-1)를 가장 가까운 클러스터에 할당
    """
    labels = labels.copy()
    noise_mask = labels == -1
    noise_count = noise_mask.sum()
    
    if noise_count == 0:
        print("No noise points to assign")
        return labels
    
    # 클러스터 ID 목록 (노이즈 제외)
    unique_clusters = np.unique(labels[~noise_mask])
    
    if len(unique_clusters) == 0:
        print("No clusters found, cannot assign noise")
        return labels
    
    # 각 클러스터의 중심점 계산
    centroids = {}
    for c in unique_clusters:
        mask = labels == c
        centroids[c] = xy[mask].mean(axis=0)
    
    centroid_ids = list(centroids.keys())
    centroid_xy = np.array([centroids[c] for c in centroid_ids])
    
    # 노이즈 포인트의 인덱스
    noise_indices = np.where(noise_mask)[0]
    
    # 각 노이즈 포인트를 가장 가까운 클러스터에 할당
    for idx in noise_indices:
        point = xy[idx]
        # 모든 중심점과의 거리 계산
        dists = np.linalg.norm(centroid_xy - point, axis=1)
        nearest_idx = np.argmin(dists)
        labels[idx] = centroid_ids[nearest_idx]
    
    print(f"Assigned {noise_count} noise points to nearest clusters")
    return labels


def relabel_clusters(labels: np.ndarray) -> tuple:
    """
    클러스터 라벨을 0부터 연속된 숫자로 재할당
    노이즈(-1)는 마지막 ID로
    """
    unique = np.unique(labels)
    has_noise = -1 in unique
    
    # 노이즈 제외한 클러스터들
    clusters = sorted([c for c in unique if c >= 0])
    
    # 새 라벨 매핑
    old_to_new = {old: new for new, old in enumerate(clusters)}
    
    # 노이즈는 마지막 ID
    noise_id = len(clusters)
    if has_noise:
        old_to_new[-1] = noise_id
    
    # 변환
    new_labels = np.array([old_to_new[l] for l in labels], dtype=np.uint16)
    
    return new_labels, len(clusters), noise_id

def align_labels_to_packed_ids(
    umap_ids: np.ndarray,
    labels_u16_umap_order: np.ndarray,
    packed_ids_path: str,
    noise_bucket: int,
) -> tuple[np.ndarray, int]:
    """
    UMAP 행 순서로 된 labels_u16을 packed_ids 순서로 재정렬.
    - umap_ids: UMAP parquet의 id 배열 (len = n_umap)
    - labels_u16_umap_order: UMAP 순서의 uint16 labels (len = n_umap)
    - packed_ids_path: 06_pack_points.py가 만든 ids_*.npy
    - noise_bucket: uint16 상에서 noise bucket id (마지막)
    반환: (labels_u16_packed_order, n_packed)
    """
    packed_ids = np.load(packed_ids_path)

    # dtype 통일(딕셔너리 키 안정화용)
    umap_ids_i = umap_ids.astype(np.int64, copy=False)
    packed_ids_i = packed_ids.astype(np.int64, copy=False)

    # id -> index (UMAP order)
    # id가 유일하다는 가정. (중복이면 마지막 값으로 덮임)
    id2idx = {int(i): idx for idx, i in enumerate(umap_ids_i)}

    out = np.empty(len(packed_ids_i), dtype=np.uint16)

    missing = 0
    for j, pid in enumerate(packed_ids_i):
        idx = id2idx.get(int(pid))
        if idx is None:
            out[j] = np.uint16(noise_bucket)
            missing += 1
        else:
            out[j] = labels_u16_umap_order[idx]

    if missing:
        print(f"[WARN] {missing} packed ids were not found in UMAP ids. "
              f"Assigned them to noise_bucket={noise_bucket}.")

    return out, len(packed_ids_i)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # UMAP 로드
    print(f"Loading UMAP: {args.umap}")
    df = pd.read_parquet(args.umap)
    
    if not {"id", "x", "y"}.issubset(df.columns):
        raise ValueError("UMAP parquet must have columns: id, x, y")
    
    n = len(df)
    print(f"Loaded {n} points from UMAP")
    
    xy = df[["x", "y"]].values.astype(np.float32)
    ids = df["id"].values
    
    # HDBSCAN 실행
    print(f"Running HDBSCAN: min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples}, method={args.cluster_selection_method}, "
          f"allow_single_cluster={args.allow_single_cluster}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        allow_single_cluster=args.allow_single_cluster,
        core_dist_n_jobs=1,
    )
    
    labels = clusterer.fit_predict(xy)
    
    # 초기 결과
    n_clusters_raw = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_raw = (labels == -1).sum()
    print(f"Initial: Clusters={n_clusters_raw}, Noise={n_noise_raw} ({100*n_noise_raw/n:.1f}%)")
    
    # 노이즈 할당 (옵션)
    if args.assign_noise and n_noise_raw > 0:
        labels = assign_noise_to_nearest(xy, labels)
    
    # 라벨 재정렬
    labels_u16, num_clusters, noise_bucket = relabel_clusters(labels)
    
    # 최종 결과
    n_noise_final = (labels_u16 == noise_bucket).sum()
    print(f"Final: Clusters={num_clusters}, Noise={n_noise_final} ({100*n_noise_final/n:.1f}%)")
    
    # 클러스터 크기 통계
    cluster_sizes = {}
    for c in range(num_clusters + 1):
        size = (labels_u16 == c).sum()
        cluster_sizes[c] = int(size)
    
    # 가장 큰 클러스터 확인
    max_cluster = max(cluster_sizes.items(), key=lambda x: x[1] if x[0] != noise_bucket else 0)
    print(f"Largest cluster: #{max_cluster[0]} with {max_cluster[1]} points ({100*max_cluster[1]/n:.1f}%)")
    
    # 저장: cluster uint16 (UMAP 순서 or packed ids 순서)
    out_cluster = out_dir / f"cluster_{args.tag}.uint16"

    if args.packed_ids_npy:
        print(f"Aligning cluster labels to packed ids: {args.packed_ids_npy}")
        labels_out, n_out = align_labels_to_packed_ids(
            umap_ids=ids,
            labels_u16_umap_order=labels_u16,
            packed_ids_path=args.packed_ids_npy,
            noise_bucket=noise_bucket,
        )
        labels_out.tofile(out_cluster)
        print(f"Saved (packed order, n={n_out}): {out_cluster}")

        # meta에서 total_points도 packed 기준으로 맞춤
        n_for_meta = n_out
        n_noise_for_meta = int((labels_out == noise_bucket).sum())
    else:
        labels_u16.tofile(out_cluster)
        print(f"Saved (umap order, n={n}): {out_cluster}")

        n_for_meta = n
        n_noise_for_meta = int(n_noise_final)

    
    # 저장: cluster labels JSON
    out_json = out_dir / f"cluster_labels_{args.tag}.json"
    meta = {
        "num_clusters": num_clusters,
        "noise_bucket": noise_bucket,
        "total_points": int(n_for_meta),
        "noise_points": int(n_noise_for_meta),
        "labels": {}  # 11_make_cluster_labels.py에서 채움
    }
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()