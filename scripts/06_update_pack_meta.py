#!/usr/bin/env python
# -*- coding: utf-8 -*-

#클러스터링 후에 실행하는 코드
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Patch pack_meta_*.json to include topic cluster files.")
    p.add_argument("--pack_meta", required=True, help="Path to pack_meta_*.json")
    p.add_argument("--tag", required=True, help="Tag like n10000_seed42")
    return p.parse_args()


def main():
    args = parse_args()
    meta_path = Path(args.pack_meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    base_dir = meta_path.parent 
    cluster_path = (base_dir / f"cluster_{args.tag}.uint16").resolve()
    labels_path = (base_dir / f"cluster_labels_{args.tag}.json").resolve()

    meta.setdefault("files", {})
    meta["files"]["cluster"] = str(cluster_path)
    meta["files"]["cluster_labels"] = str(labels_path)

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Patched:", meta_path)
    print(" - cluster:", meta['files']['cluster'])
    print(" - cluster_labels:", meta['files']['cluster_labels'])


if __name__ == "__main__":
    main()