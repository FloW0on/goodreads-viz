#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Pack UMAP 2D points (id,x,y) into binary buffers for WebGPU."
    )
    p.add_argument("--umap", required=True, help="UMAP parquet with columns: id, x, y")
    p.add_argument("--meta", default=None, help="Optional meta parquet with column: id (join key)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", default="", help="Suffix tag for output filenames")
    p.add_argument("--center", action="store_true", help="Center x,y by subtracting mean")
    p.add_argument("--scale", type=float, default=1.0, help="Multiply x,y by this factor")
    p.add_argument("--id_dtype", default="uint32", choices=["uint32", "int64"], help="ID dtype for binary output")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.umap)
    need = {"id", "x", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"UMAP parquet missing columns: {miss}")

    # Optional join with meta
    if args.meta:
        meta = pd.read_parquet(args.meta).drop_duplicates("id")
        df = df.merge(meta, on="id", how="left")

    # Drop duplicates and stabilize ordering
    df = df.drop_duplicates("id").sort_values("id").reset_index(drop=True)

    # Extract xy
    x = df["x"].astype("float32").to_numpy()
    y = df["y"].astype("float32").to_numpy()

    if args.center:
        x = x - x.mean()
        y = y - y.mean()

    x *= float(args.scale)
    y *= float(args.scale)

    points = np.stack([x, y], axis=1).astype("float32")  # (N,2)

    # IDs
    if args.id_dtype == "uint32":
        # Your ids max is ~3.6e7, safely fits uint32
        ids = df["id"].astype("uint32").to_numpy()
        ids_dtype = "uint32"
    else:
        ids = df["id"].astype("int64").to_numpy()
        ids_dtype = "int64"

    tag = f"_{args.tag}" if args.tag else ""

    points_bin = out_dir / f"points_xy{tag}.f32"
    ids_bin = out_dir / f"ids{tag}.{ids_dtype}"
    debug_parquet = out_dir / f"points_debug{tag}.parquet"
    meta_json = out_dir / f"pack_meta{tag}.json"

    points.tofile(points_bin)
    ids.tofile(ids_bin)
    df.to_parquet(debug_parquet, index=False)

    meta = {
        "n": int(len(df)),
        "xy": {
            "dtype": "float32",
            "shape": [int(len(df)), 2],
            "layout": "row-major Nx2 (x,y)",
            "center": bool(args.center),
            "scale": float(args.scale),
        },
        "ids": {
            "dtype": ids_dtype,
            "shape": [int(len(df))],
        },
        "columns_in_debug_parquet": df.columns.tolist(),
        "files": {
            "points_xy": str(points_bin),
            "ids": str(ids_bin),
            "debug_parquet": str(debug_parquet),
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(" -", points_bin, f"({points.nbytes/1024/1024:.2f} MiB)")
    print(" -", ids_bin, f"({ids.nbytes/1024/1024:.2f} MiB)")
    print(" -", debug_parquet)
    print(" -", meta_json)


if __name__ == "__main__":
    main()