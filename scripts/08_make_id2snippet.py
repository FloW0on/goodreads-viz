#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser("Make id->text snippet json")
    p.add_argument("--texts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_chars", type=int, default=160)
    return p.parse_args()

def main():
    a = parse_args()
    df = pd.read_parquet(a.texts)[["id", "text_for_embed"]]

    out = {}
    for _, r in df.iterrows():
        text = str(r["text_for_embed"]).replace("\n", " ").strip()
        out[str(int(r["id"]))] = text[:a.max_chars]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("Saved:", a.out, "n=", len(out))

if __name__ == "__main__":
    main()