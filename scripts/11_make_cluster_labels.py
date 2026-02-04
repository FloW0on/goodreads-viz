#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_make_cluster_labels.py
Contrastive KeyBERT - 클러스터 vs 전체 대비로 차별화된 키워드 추출
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

CUSTOM_STOP = {
  "standalone","excerpt","excerpts","hardcover","paperback","kindle",
  "series","trilogy","book","novel","story","author","read","reading",
}

def is_bad_kw(kw: str) -> bool:
    k = kw.strip()
    if len(k) < 4: 
        return True
    if any(ch.isdigit() for ch in k):
        return True
    if re.search(r"[^a-zA-Z\s\-']", k):
        return True
    toks = k.split()
    # 두 토큰 이상에서 모두 대문자 시작이면 고유명사 가능성 큼
    if len(toks) >= 2 and all(t[0].isupper() for t in toks if t):
        return True
    if k.lower() in CUSTOM_STOP:
        return True
    return False

def parse_args():
    p = argparse.ArgumentParser(description="Generate cluster labels with Contrastive KeyBERT")
    p.add_argument("--ids_npy", required=True, help="Packed IDs .npy file")
    p.add_argument("--texts", required=True, help="Texts parquet (id, text_for_embed)")
    p.add_argument("--cluster_uint16", required=True, help="Cluster assignments .uint16 file")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", required=True, help="Output tag")
    p.add_argument("--topk", type=int, default=5, help="Top keywords per cluster")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    p.add_argument("--sample_per_cluster", type=int, default=50, 
                   help="Representative docs to sample per cluster")
    p.add_argument("--ngram_range", type=str, default="1,2", 
                   help="N-gram range for keyphrases")
    return p.parse_args()


def get_representative_docs(docs, n_samples=50):
    """랜덤 샘플링으로 대표 문서 선택"""
    if len(docs) <= n_samples:
        return docs
    
    return random.sample(docs, n_samples)


def extract_contrastive_keywords(
    cluster_docs,
    global_doc_sample,
    kw_model,
    topk=5,
    ngram_range=(1, 2),
    sample_per_cluster=50
):
    """
    Contrastive 키워드 추출
    - 각 클러스터의 대표 문서에서 후보 키워드 추출
    - 전체 corpus 대비 해당 클러스터에서 더 특징적인 키워드 선택
    """
    
    # 1. 전체 corpus 임베딩 (비교용)
    print("  Embedding global sample...")
    global_text = " ".join(global_doc_sample[:200])  # 전체 대표 샘플
    
    # 2. 전체에서 자주 나오는 키워드 추출 (제외할 것들)
    global_keywords = kw_model.extract_keywords(
        global_text,
        keyphrase_ngram_range=ngram_range,
        stop_words='english',
        top_n=300
    )
    global_kw_set = {kw.lower() for kw, score in global_keywords}
    
    results = {}
    
    for cid, docs in tqdm(cluster_docs.items(), desc="  Extracting keywords"):
        if not docs:
            results[cid] = []
            continue
        
        # 대표 문서 선택 (랜덤 샘플링)
        rep_docs = get_representative_docs(docs, sample_per_cluster)
        cluster_text = " ".join(rep_docs)
        
        if len(cluster_text) > 50000:
            cluster_text = cluster_text[:50000]
        
        try:
            # 클러스터에서 키워드 추출 (많이 추출)
            cluster_keywords = kw_model.extract_keywords(
                cluster_text,
                keyphrase_ngram_range=ngram_range,
                stop_words='english',
                use_mmr=True,
                diversity=0.5,
                top_n=topk * 5
            )
            
            # Contrastive 필터링: 전체에서 흔한 키워드 제외
            contrastive_keywords = []
            for kw, score in cluster_keywords:
                if is_bad_kw(kw): 
                    continue
                kw_lower = kw.lower()
                # 전체 corpus에서 상위 키워드가 아닌 것만 선택
                if kw_lower not in global_kw_set:
                    contrastive_keywords.append((kw, score))
                    if len(contrastive_keywords) >= topk:
                        break
            
            # 부족하면 원래 키워드에서 채움
            if len(contrastive_keywords) < topk:
                for kw, score in cluster_keywords:
                    if kw not in [k for k, s in contrastive_keywords]:
                        contrastive_keywords.append((kw, score))
                        if len(contrastive_keywords) >= topk:
                            break
            
            results[cid] = contrastive_keywords
            
        except Exception as e:
            print(f"    Warning: Cluster {cid} failed: {e}")
            results[cid] = []
    
    return results


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    
    ngram_min, ngram_max = map(int, args.ngram_range.split(","))
    
    # 데이터 로드
    print(f"Loading IDs: {args.ids_npy}")
    ids = np.load(args.ids_npy)
    
    print(f"Loading texts: {args.texts}")
    texts_df = pd.read_parquet(args.texts)
    
    print(f"Loading clusters: {args.cluster_uint16}")
    clusters = np.fromfile(args.cluster_uint16, dtype=np.uint16)
    
    n = len(ids)
    print(f"Points: {n}")
    
    # ID → 텍스트 매핑
    id_col = 'id' if 'id' in texts_df.columns else texts_df.columns[0]
    text_col = None
    for col in ['text_for_embed', 'text', 'title', 'description']:
        if col in texts_df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = [c for c in texts_df.columns if c != id_col][0]
    
    print(f"Using text column: {text_col}")
    
    id_to_text = {}
    all_texts = []
    for _, row in texts_df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        id_to_text[int(row[id_col])] = text
        if len(text) > 50:
            all_texts.append(text)
    
    # 클러스터별 문서 수집
    cluster_docs = defaultdict(list)
    cluster_sizes = Counter()
    
    for book_id, cluster_id in zip(ids, clusters):
        cluster_id = int(cluster_id)
        cluster_sizes[cluster_id] += 1
        
        text = id_to_text.get(int(book_id), "")
        if text and len(text) > 20:
            cluster_docs[cluster_id].append(text)
    
    num_clusters = len(cluster_sizes)
    print(f"Clusters: {num_clusters}")
    
    # 전체 corpus 샘플 (contrastive 비교용)
    random.seed(42)
    global_sample = random.sample(all_texts, min(2000, len(all_texts)))
    print(f"Global sample size: {len(global_sample)}")
    
    # 모델 초기화
    print(f"\nInitializing KeyBERT...")
    from keybert import KeyBERT
    
    kw_model = KeyBERT(model=args.model)
    
    # Contrastive 키워드 추출
    print(f"\nExtracting contrastive keywords...")
    cluster_keywords = extract_contrastive_keywords(
        cluster_docs,
        global_sample,
        kw_model,
        topk=args.topk,
        ngram_range=(ngram_min, ngram_max),
        sample_per_cluster=args.sample_per_cluster
    )
    
    # 결과 저장
    labels = {}
    for cid in sorted(cluster_sizes.keys()):
        keywords = cluster_keywords.get(cid, [])
        keyword_list = [kw for kw, score in keywords]
        label_str = " / ".join(keyword_list) if keyword_list else f"cluster_{cid}"
        
        labels[str(cid)] = {
            "keywords": keyword_list,
            "label": label_str,
            "size": cluster_sizes[cid],
            "top_words": [{"word": kw, "score": round(score, 4)} for kw, score in keywords]
        }
    
    # JSON 저장
    out_json = out_dir / f"cluster_labels_{args.tag}.json"
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        existing = {}
    
    existing["labels"] = labels
    existing["num_clusters"] = num_clusters
    existing["noise_bucket"] = int(clusters.max())
    
    out_json.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")
    
    # 샘플 출력
    print("\n=== Sample Labels (Top 10 by size) ===")
    sorted_by_size = sorted(cluster_sizes.items(), key=lambda x: -x[1])[:10]
    for cid, size in sorted_by_size:
        label = labels[str(cid)]["label"]
        print(f"  #{cid} ({size:,} pts): {label}")


if __name__ == "__main__":
    main()