#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Create cluster-based circular layout (packed-order safe)")
    p.add_argument("--umap", required=True, help="UMAP parquet with columns: id, x, y")
    p.add_argument("--cluster", required=True, help="Cluster labels .uint16 file (packed order)")
    p.add_argument("--packed_ids_npy", required=True, help="Packed ids .npy (point order)")
    p.add_argument("--out_dir", required=True, help="Output directory (packed)")
    p.add_argument("--tag", required=True, help="Output tag")

    p.add_argument("--cluster_radius", type=float, default=1.0, help="Base radius of each cluster circle")
    p.add_argument("--layout_radius", type=float, default=3.0, help="Radius of ring for cluster centers")
    p.add_argument("--noise_center", action="store_true", help="Place noise bucket at center")
    p.add_argument("--preserve_internal", action="store_true", help="Preserve relative positions within each cluster")

    # optional integration
    p.add_argument("--patch_pack_meta", action="store_true",
                   help="Patch pack_meta_<tag>.json to include layout file path")

    return p.parse_args()

def preserve_internal_layout(points_xy: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Preserve cluster internal geometry by min-max normalizing each axis to [-1,1],
    then scaling into radius (with margin), then translating to center.
    """
    n = points_xy.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return center.reshape(1, 2).astype(np.float32)

    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy = np.where(range_xy == 0, 1.0, range_xy)

    normalized = 2.0 * (points_xy - min_xy) / range_xy - 1.0  # [-1,1]
    scaled = normalized * (radius * 0.8)  # leave margin
    out = scaled + center.reshape(1, 2)
    return out.astype(np.float32)


def patch_pack_meta(out_dir: Path, tag: str, layout_xy_path: Path, layout_meta_path: Path):
    meta_path = out_dir / f"pack_meta_{tag}.json"
    if not meta_path.exists():
        print("[patch] pack_meta not found:", meta_path)
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("files", {})
    meta["files"]["points_xy_layout"] = str(layout_xy_path.resolve())
    meta["files"]["layout_meta"] = str(layout_meta_path.resolve())
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[patch] updated:", meta_path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load packed ids (point order)
    packed_ids = np.load(args.packed_ids_npy)
    if packed_ids.ndim != 1:
        packed_ids = packed_ids.reshape(-1)
    packed_ids = packed_ids.astype(np.int64, copy=False)
    n = int(packed_ids.shape[0])

    # 2) Load UMAP parquet and align to packed order
    print(f"Loading UMAP: {args.umap}")
    dfu = pd.read_parquet(args.umap)
    need = {"id", "x", "y"}
    if not need.issubset(dfu.columns):
        raise ValueError(f"UMAP parquet must contain columns {sorted(need)}; got {dfu.columns.tolist()}")

    dfu = dfu[["id", "x", "y"]].copy()
    dfu["id"] = dfu["id"].astype(np.int64)

    if dfu.shape[0] != n:
        raise ValueError(f"UMAP rows ({dfu.shape[0]}) != packed_ids length ({n}). "
                         f"Use the same tag/run outputs.")

    # fast id->row mapping
    id_to_row = {int(bid): i for i, bid in enumerate(dfu["id"].to_list())}

    rows = np.empty(n, dtype=np.int64)
    missing = 0
    for i, bid in enumerate(packed_ids.tolist()):
        r = id_to_row.get(int(bid), -1)
        if r < 0:
            missing += 1
        rows[i] = r
    if missing:
        ex = [int(bid) for bid, r in zip(packed_ids[:2000].tolist(), rows[:2000].tolist()) if r < 0][:10]
        raise KeyError(f"Missing {missing} packed ids in UMAP parquet. Examples: {ex}")

    original_xy = dfu.loc[rows, ["x", "y"]].to_numpy(dtype=np.float32)  # packed order
    ids_in_order = packed_ids  # packed order

    # 3) Load cluster labels (must be packed order)
    print(f"Loading clusters: {args.cluster}")
    clusters = np.fromfile(args.cluster, dtype=np.uint16)
    if clusters.shape[0] != n:
        raise ValueError(f"Mismatch: packed_ids has {n} points, clusters has {clusters.shape[0]}")

    unique_clusters = np.unique(clusters)
    print(f"Points: {n}")
    print(f"Unique cluster ids: {unique_clusters.tolist()}")

    # cluster sizes
    cluster_sizes = {int(c): int((clusters == c).sum()) for c in unique_clusters}
    total_points = int(sum(cluster_sizes.values()))
    print("Cluster sizes:", cluster_sizes)

    # 4) Decide noise bucket (by convention: max id)
    noise_cluster = int(unique_clusters.max())
    non_noise = [int(c) for c in unique_clusters.tolist() if int(c) != noise_cluster]
    num_main = len(non_noise)

    # 5) Assign cluster centers from original UMAP centroids (NO RING)
    cluster_centers = {}
    cluster_radii = {}

    # radii는 클러스터 크기에 비례 (겹침 방지용)
    for c in non_noise:
        idx = np.where(clusters == c)[0]
        center = original_xy[idx].mean(axis=0).astype(np.float32) 
        cluster_centers[int(c)] = center

        size_ratio = cluster_sizes[int(c)] / max(total_points, 1)
        cluster_radii[int(c)] = float(args.cluster_radius * (0.5 + size_ratio * 2.0))

    # noise center placement
    if args.noise_center:
        cluster_centers[int(noise_cluster)] = np.array([0.0, 0.0], dtype=np.float32)
    else:
        # noise도 centroid 기반으로
        idx = np.where(clusters == noise_cluster)[0]
        cluster_centers[int(noise_cluster)] = original_xy[idx].mean(axis=0).astype(np.float32)

    cluster_radii[int(noise_cluster)] = float(args.cluster_radius * 0.5)

    # small repulsion between cluster centers to reduce overlap
    PUSH = 0.15 

    main = [int(c) for c in non_noise]
    for i in range(len(main)):
        for j in range(i + 1, len(main)):
            c1, c2 = main[i], main[j]
            v = cluster_centers[c1] - cluster_centers[c2]
            d = float(np.linalg.norm(v))
            if d < 1e-6:
                continue
            shift = (PUSH / d) * v.astype(np.float32)
            cluster_centers[c1] += shift
            cluster_centers[c2] -= shift

    # 6) Build new XY (packed order)
    new_xy = np.zeros((n, 2), dtype=np.float32)

    for c in unique_clusters.tolist():
        c = int(c)
        idx = np.where(clusters == c)[0]
        center = cluster_centers[c]
        radius = cluster_radii[c]

        pts = original_xy[idx]
        new_pts = preserve_internal_layout(pts, center, radius)
        new_xy[idx] = new_pts

    # 7) Save outputs
    out_parquet = out_dir / f"layout_{args.tag}.parquet"
    df_out = pd.DataFrame({"id": ids_in_order.astype(np.int64), "x": new_xy[:, 0], "y": new_xy[:, 1]})
    df_out.to_parquet(out_parquet, index=False)

    out_xy = out_dir / f"points_xy_{args.tag}_layout.f32"
    new_xy.astype(np.float32).tofile(out_xy)

    out_meta = out_dir / f"layout_{args.tag}.json"
    meta = {
        "tag": args.tag,
        "n": int(n),
        "layout": "cluster_circular",
        "params": {
            "cluster_radius": float(args.cluster_radius),
            "layout_radius": float(args.layout_radius),
            "noise_center": bool(args.noise_center),
            "preserve_internal": bool(args.preserve_internal),
        },
        "noise_cluster": int(noise_cluster),
        "cluster_info": {
            str(int(c)): {
                "center": cluster_centers[int(c)].tolist(),
                "radius": float(cluster_radii[int(c)]),
                "size": int(cluster_sizes[int(c)]),
            }
            for c in unique_clusters.tolist()
        },
        "files": {
            "layout_parquet": str(out_parquet.resolve()),
            "points_xy_layout": str(out_xy.resolve()),
            "packed_ids_npy": str(Path(args.packed_ids_npy).resolve()),
            "cluster_uint16": str(Path(args.cluster).resolve()),
        },
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", out_parquet)
    print(" -", out_xy)
    print(" -", out_meta)

    if args.patch_pack_meta:
        patch_pack_meta(out_dir, args.tag, out_xy, out_meta)

    print("\nDone!")


if __name__ == "__main__":
    main()