#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_assign_ddc.py
책 임베딩을 DDC 10개 주류에 할당 (미분류 강제 할당 옵션 포함)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# DDC 10개 주류 정의
DDC_CLASSES = {
    0: {"name": "총류 (컴퓨터, 정보)", "name_ko": "총류 (컴퓨터, 정보)",
        "keywords": "computer science, information, encyclopedia, library, journalism, publishing"},
    1: {"name": "철학, 심리학", "name_ko": "철학, 심리학",
        "keywords": "philosophy, psychology, ethics, logic, metaphysics, epistemology, mind, consciousness"},
    2: {"name": "종교", "name_ko": "종교",
        "keywords": "religion, bible, christianity, islam, buddhism, hinduism, spirituality, theology"},
    3: {"name": "사회과학", "name_ko": "사회과학",
        "keywords": "sociology, politics, economics, law, education, government, commerce, social issues"},
    4: {"name": "언어", "name_ko": "언어",
        "keywords": "language, linguistics, grammar, dictionary, english, french, german, spanish"},
    5: {"name": "자연과학", "name_ko": "자연과학",
        "keywords": "science, mathematics, physics, chemistry, biology, astronomy, geology, nature"},
    6: {"name": "기술, 응용과학", "name_ko": "기술, 응용과학",
        "keywords": "technology, medicine, engineering, agriculture, cooking, business, health, medical"},
    7: {"name": "예술, 오락", "name_ko": "예술, 오락",
        "keywords": "art, music, sports, games, entertainment, painting, sculpture, photography, film"},
    8: {"name": "문학", "name_ko": "문학",
        "keywords": "literature, fiction, poetry, drama, novel, story, prose, literary criticism"},
    9: {"name": "역사, 지리", "name_ko": "역사, 지리",
        "keywords": "history, geography, biography, travel, archaeology, ancient, world war, civilization"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Assign DDC classes to book embeddings")
    p.add_argument("--embeddings_npy", required=True, help="Book embeddings .npy file")
    p.add_argument("--emb_ids_npy", required=True, help="Embedding IDs .npy file")
    p.add_argument("--packed_ids_npy", required=True, help="Packed IDs .npy file (target order)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", required=True, help="Output tag")
    
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--confidence_threshold", type=float, default=0.1,
                   help="Minimum similarity score to assign a DDC class")
    
    # 강제 할당 옵션
    p.add_argument("--force_assign", action="store_true",
                   help="Force assign all points to DDC (no Unknown)")
    
    p.add_argument("--patch_pack_meta", action="store_true",
                   help="Patch pack_meta JSON with DDC file paths")
    
    # memmap shape 파라미터
    p.add_argument("--n_records", type=int, default=500000, help="Number of records in embeddings")
    p.add_argument("--emb_dim", type=int, default=384, help="Embedding dimension")
    
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 임베딩 로드 (memmap)
    embeddings = np.memmap(args.embeddings_npy, dtype="float32", mode="r", shape=(args.n_records, args.emb_dim))
    emb_ids = np.memmap(args.emb_ids_npy, dtype="int64", mode="r", shape=(args.n_records,))
    packed_ids = np.load(args.packed_ids_npy)
    
    print(f"Embeddings: {embeddings.shape} dtype={embeddings.dtype}")
    print(f"emb_ids: {emb_ids.shape} dtype={emb_ids.dtype}")
    print(f"packed_ids: {packed_ids.shape} dtype={packed_ids.dtype}")
    
    n = len(packed_ids)
    emb_dim = embeddings.shape[1]
    
    # ID -> index 매핑
    emb_id_to_idx = {int(eid): i for i, eid in enumerate(emb_ids)}
    
    # 2. 모델 로드
    print(f"Loading model: {args.model} ({args.device})")
    model = SentenceTransformer(args.model, device=args.device)
    
    # 3. DDC 프로토타입 임베딩 생성
    from tqdm import tqdm
    
    ddc_texts = [DDC_CLASSES[i]["keywords"] for i in range(10)]
    ddc_embeddings = []
    
    for text in tqdm(ddc_texts, desc="Embedding DDC prototypes"):
        emb = model.encode(text, convert_to_numpy=True)
        ddc_embeddings.append(emb)
    
    ddc_embeddings = np.array(ddc_embeddings, dtype=np.float32)  # (10, dim)
    
    # 정규화
    ddc_norms = np.linalg.norm(ddc_embeddings, axis=1, keepdims=True)
    ddc_embeddings = ddc_embeddings / (ddc_norms + 1e-8)
    
    # 4. 각 책에 DDC 할당
    ddc_assignments = np.zeros(n, dtype=np.uint16)
    scores_list = []
    
    for i, pid in enumerate(tqdm(packed_ids, desc="Assigning DDC")):
        pid = int(pid)
        idx = emb_id_to_idx.get(pid, -1)
        
        if idx < 0:
            # 임베딩 없음 → Unknown
            ddc_assignments[i] = 10
            scores_list.append(-1.0)
            continue
        
        # 코사인 유사도 계산
        book_emb = embeddings[idx]
        book_norm = np.linalg.norm(book_emb)
        if book_norm < 1e-8:
            ddc_assignments[i] = 10
            scores_list.append(-1.0)
            continue
        
        book_emb = book_emb / book_norm
        similarities = ddc_embeddings @ book_emb  # (10,)
        
        best_ddc = int(np.argmax(similarities))
        best_score = float(similarities[best_ddc])
        
        # 강제 할당 모드: threshold 무시
        if args.force_assign:
            ddc_assignments[i] = best_ddc
        else:
            # threshold 적용
            if best_score >= args.confidence_threshold:
                ddc_assignments[i] = best_ddc
            else:
                ddc_assignments[i] = 10  # Unknown
        
        scores_list.append(best_score)
    
    scores_arr = np.array(scores_list)
    
    # 5. 통계 출력
    print("=" * 50)
    print("DDC Assignment Statistics")
    print("=" * 50)
    
    for i in range(11):
        count = (ddc_assignments == i).sum()
        pct = 100 * count / n
        if i < 10:
            name = DDC_CLASSES[i]["name_ko"]
            print(f"  {i}: {name:20s} | {count:7d} ({pct:5.1f}%)")
        else:
            print(f"  {i}: {'Unknown':20s} | {count:7d} ({pct:5.1f}%)")
    
    valid_scores = scores_arr[scores_arr >= 0]
    if len(valid_scores) > 0:
        print(f"Scores: min={valid_scores.min():.3f}, max={valid_scores.max():.3f}, mean={valid_scores.mean():.3f}")
    
    # 6. 저장
    out_ddc = out_dir / f"ddc_{args.tag}.uint16"
    ddc_assignments.tofile(out_ddc)
    
    out_meta = out_dir / f"ddc_meta_{args.tag}.json"
    meta = {
        "num_classes": 10,
        "unknown_id": 10,
        "threshold": args.confidence_threshold if not args.force_assign else None,
        "force_assign": args.force_assign,
        "model": args.model,
        "classes": {
            str(i): {
                "name": DDC_CLASSES[i]["name"],
                "name_ko": DDC_CLASSES[i]["name_ko"],
                "count": int((ddc_assignments == i).sum()),
            }
            for i in range(10)
        },
        "unknown_count": int((ddc_assignments == 10).sum()),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"Saved:")
    print(f" - {out_ddc}")
    print(f" - {out_meta}")
    
    # 7. pack_meta 패치
    if args.patch_pack_meta:
        pack_meta_path = out_dir / f"pack_meta_{args.tag}.json"
        if pack_meta_path.exists():
            pack_meta = json.loads(pack_meta_path.read_text(encoding="utf-8"))
            pack_meta.setdefault("files", {})
            pack_meta["files"]["ddc"] = str(out_ddc.resolve())
            pack_meta["files"]["ddc_meta"] = str(out_meta.resolve())
            pack_meta_path.write_text(json.dumps(pack_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[patch] updated: {pack_meta_path}")


if __name__ == "__main__":
    main()