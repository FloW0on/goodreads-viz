#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
클러스터 기반 원형 레이아웃 생성

기존 UMAP 좌표와 클러스터 라벨을 사용해서
각 클러스터가 시각적으로 분리된 원형 그룹으로 보이도록 재배치

사용법:
    python 13_cluster_layout.py \
        --umap ./umap2d.parquet \
        --cluster ./cluster.uint16 \
        --out_dir ./web/public/packed \
        --tag n10000_seed42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Create cluster-based circular layout")
    p.add_argument("--umap", required=True, help="UMAP parquet with columns: id, x, y")
    p.add_argument("--cluster", required=True, help="Cluster labels .uint16 file")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", required=True, help="Output tag")
    p.add_argument("--cluster_radius", type=float, default=1.0, 
                   help="Radius of each cluster circle")
    p.add_argument("--layout_radius", type=float, default=3.0,
                   help="Radius of the overall layout (distance from center to cluster centers)")
    p.add_argument("--noise_center", action="store_true",
                   help="Place noise cluster at center (default: on the ring)")
    p.add_argument("--preserve_internal", action="store_true",
                   help="Preserve relative positions within each cluster")
    return p.parse_args()


def normalize_to_circle(points, radius=1.0):
    """
    포인트들을 원 안에 균등하게 배치
    """
    n = len(points)
    if n == 0:
        return np.array([]).reshape(0, 2)
    
    if n == 1:
        return np.array([[0.0, 0.0]])
    
    # Sunflower 패턴으로 원 안에 균등 배치
    indices = np.arange(n) + 0.5
    r = radius * np.sqrt(indices / n)
    theta = np.pi * (1 + np.sqrt(5)) * indices  # Golden angle
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack([x, y])


def preserve_internal_layout(points, center, radius):
    """
    클러스터 내 상대적 위치를 유지하면서 원 안에 맞춤
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    
    if len(points) == 1:
        return np.array([center])
    
    # 현재 범위 계산
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    range_xy = max_xy - min_xy
    
    # 0으로 나누기 방지
    range_xy = np.where(range_xy == 0, 1, range_xy)
    
    # 정규화 (-1 ~ 1)
    normalized = 2 * (points - min_xy) / range_xy - 1
    
    # 반지름에 맞게 스케일
    scaled = normalized * radius * 0.8  # 0.8로 여백
    
    # 중심 이동
    return scaled + center


def main():
    args = parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print(f"Loading UMAP: {args.umap}")
    df = pd.read_parquet(args.umap)
    
    print(f"Loading clusters: {args.cluster}")
    clusters = np.fromfile(args.cluster, dtype=np.uint16)
    
    n = len(df)
    if len(clusters) != n:
        raise ValueError(f"Mismatch: UMAP has {n} points, clusters has {len(clusters)}")
    
    print(f"Points: {n}")
    
    # 클러스터 정보
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)
    print(f"Clusters: {unique_clusters.tolist()}")
    
    # 클러스터별 크기 계산 (레이아웃 배치용)
    cluster_sizes = {c: (clusters == c).sum() for c in unique_clusters}
    print("Cluster sizes:", cluster_sizes)
    
    # 클러스터 중심 위치 계산 (원형 배치)
    # 크기가 큰 클러스터일수록 반지름도 크게
    total_points = sum(cluster_sizes.values())
    
    cluster_centers = {}
    cluster_radii = {}
    
    # noise 클러스터 (보통 마지막)
    noise_cluster = max(unique_clusters)
    non_noise_clusters = [c for c in unique_clusters if c != noise_cluster]
    
    # 클러스터 중심을 원형으로 배치
    num_main_clusters = len(non_noise_clusters)
    
    for i, c in enumerate(non_noise_clusters):
        angle = 2 * np.pi * i / max(num_main_clusters, 1)
        cx = args.layout_radius * np.cos(angle)
        cy = args.layout_radius * np.sin(angle)
        cluster_centers[c] = np.array([cx, cy])
        
        # 클러스터 크기에 비례한 반지름
        size_ratio = cluster_sizes[c] / total_points
        cluster_radii[c] = args.cluster_radius * (0.5 + size_ratio * 2)
    
    # noise 클러스터
    if args.noise_center:
        cluster_centers[noise_cluster] = np.array([0.0, 0.0])
    else:
        # 원형 배치의 마지막에 추가
        angle = 2 * np.pi * num_main_clusters / max(num_main_clusters + 1, 1)
        cluster_centers[noise_cluster] = np.array([
            args.layout_radius * np.cos(angle),
            args.layout_radius * np.sin(angle)
        ])
    cluster_radii[noise_cluster] = args.cluster_radius * 0.5
    
    print("\nCluster centers:")
    for c in unique_clusters:
        print(f"  {c}: center={cluster_centers[c]}, radius={cluster_radii[c]:.2f}")
    
    # 새로운 좌표 생성
    new_xy = np.zeros((n, 2), dtype=np.float32)
    
    original_xy = df[["x", "y"]].to_numpy()
    
    for c in unique_clusters:
        mask = clusters == c
        indices = np.where(mask)[0]
        
        center = cluster_centers[c]
        radius = cluster_radii[c]
        
        if args.preserve_internal:
            # 클러스터 내 상대적 위치 유지
            cluster_points = original_xy[mask]
            new_points = preserve_internal_layout(cluster_points, center, radius)
        else:
            # Sunflower 패턴으로 균등 배치
            new_points = normalize_to_circle(indices, radius) + center
        
        new_xy[indices] = new_points
    
    # 결과 저장
    df_out = pd.DataFrame({
        "id": df["id"].values,
        "x": new_xy[:, 0],
        "y": new_xy[:, 1],
    })
    
    out_parquet = out_dir / f"layout_{args.tag}.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"\nSaved: {out_parquet}")
    
    # Binary 포맷도 저장 (웹용)
    out_xy = out_dir / f"points_xy_{args.tag}_layout.f32"
    new_xy.astype(np.float32).tofile(out_xy)
    print(f"Saved: {out_xy}")
    
    out_ids = out_dir / f"ids_{args.tag}_layout.uint32"
    df["id"].to_numpy().astype(np.uint32).tofile(out_ids)
    print(f"Saved: {out_ids}")
    
    # 메타데이터
    meta = {
        "tag": args.tag,
        "n": int(n),
        "layout": "cluster_circular",
        "params": {
            "cluster_radius": args.cluster_radius,
            "layout_radius": args.layout_radius,
            "preserve_internal": args.preserve_internal,
        },
        "cluster_info": {
            str(c): {
                "center": cluster_centers[c].tolist(),
                "radius": float(cluster_radii[c]),
                "size": int(cluster_sizes[c]),
            }
            for c in unique_clusters
        },
        "files": {
            "points_xy": str(out_xy),
            "ids": str(out_ids),
        }
    }
    
    out_meta = out_dir / f"layout_{args.tag}.json"
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_meta}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()