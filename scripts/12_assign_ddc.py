#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDC 10대 분류 자동 할당 (임베딩 유사도 기반)

사용법:
    python 12_assign_ddc.py \
        --snippet_json ./web/public/packed/id2snippet_n10000_seed42.json \
        --ids_file ./web/public/packed/ids_n10000_seed42_umap2d.uint32 \
        --out_dir ./web/public/packed \
        --tag n10000_seed42
"""

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ============================================================
# DDC 10대 분류 정의
# ============================================================

DDC_CLASSES = {
    0: {
        "name": "Computer Science, Information & General Works",
        "name_ko": "총류 (컴퓨터, 정보)",
        "color": "#8dd3c7",  # 민트
        "keywords": [
            "computer science", "programming", "software", "data science",
            "information technology", "artificial intelligence", "machine learning",
            "algorithms", "database", "internet", "web development",
            "encyclopedia", "library science", "journalism", "news media",
            "general knowledge", "reference", "bibliography"
        ],
        "examples": [
            "Introduction to Computer Science and Programming",
            "Data Structures and Algorithms",
            "Machine Learning: A Probabilistic Perspective",
            "The Art of Computer Programming",
            "Encyclopedia Britannica",
            "Information Architecture for the Web"
        ]
    },
    1: {
        "name": "Philosophy & Psychology",
        "name_ko": "철학, 심리학",
        "color": "#ffffb3",  # 연노랑
        "keywords": [
            "philosophy", "ethics", "logic", "metaphysics", "epistemology",
            "psychology", "psychoanalysis", "cognitive science", "consciousness",
            "mind", "behavior", "mental health", "therapy", "counseling",
            "self-help", "personal development", "motivation", "happiness",
            "stoicism", "existentialism", "phenomenology"
        ],
        "examples": [
            "Meditations by Marcus Aurelius",
            "Thinking, Fast and Slow",
            "The Interpretation of Dreams by Freud",
            "Man's Search for Meaning",
            "The Republic by Plato",
            "Cognitive Psychology: A Student's Handbook"
        ]
    },
    2: {
        "name": "Religion",
        "name_ko": "종교",
        "color": "#bebada",  # 연보라
        "keywords": [
            "religion", "theology", "bible", "christianity", "islam", "buddhism",
            "hinduism", "judaism", "spirituality", "faith", "god", "prayer",
            "church", "mosque", "temple", "scripture", "sacred", "divine",
            "meditation", "zen", "mysticism", "afterlife", "soul"
        ],
        "examples": [
            "The Bible",
            "The Quran",
            "Mere Christianity by C.S. Lewis",
            "The Power of Now: A Guide to Spiritual Enlightenment",
            "Siddhartha by Hermann Hesse",
            "The Bhagavad Gita"
        ]
    },
    3: {
        "name": "Social Sciences",
        "name_ko": "사회과학",
        "color": "#fb8072",  # 연빨강/코랄
        "keywords": [
            "sociology", "economics", "politics", "government", "law",
            "education", "commerce", "business", "management", "finance",
            "anthropology", "social issues", "poverty", "inequality",
            "democracy", "capitalism", "socialism", "policy", "statistics",
            "criminology", "welfare", "globalization", "trade"
        ],
        "examples": [
            "Capital in the Twenty-First Century",
            "Freakonomics",
            "The Wealth of Nations by Adam Smith",
            "Sapiens: A Brief History of Humankind",
            "Thinking Like an Economist",
            "Introduction to Political Science"
        ]
    },
    4: {
        "name": "Language",
        "name_ko": "언어",
        "color": "#80b1d3",  # 연파랑
        "keywords": [
            "language", "linguistics", "grammar", "vocabulary", "dictionary",
            "translation", "etymology", "phonetics", "syntax", "semantics",
            "english", "spanish", "french", "german", "chinese", "japanese",
            "language learning", "writing skills", "communication", "rhetoric"
        ],
        "examples": [
            "The Elements of Style",
            "English Grammar in Use",
            "The Language Instinct by Steven Pinker",
            "Fluent Forever: How to Learn Any Language",
            "Oxford English Dictionary",
            "Introduction to Linguistics"
        ]
    },
    5: {
        "name": "Science",
        "name_ko": "자연과학",
        "color": "#fdb462",  # 연주황
        "keywords": [
            "science", "physics", "chemistry", "biology", "mathematics",
            "astronomy", "geology", "ecology", "evolution", "genetics",
            "quantum", "relativity", "atom", "molecule", "cell", "organism",
            "experiment", "theory", "hypothesis", "scientific method",
            "nature", "universe", "cosmos", "planet", "species"
        ],
        "examples": [
            "A Brief History of Time by Stephen Hawking",
            "The Origin of Species by Charles Darwin",
            "Cosmos by Carl Sagan",
            "The Selfish Gene by Richard Dawkins",
            "Principles of Physics",
            "Organic Chemistry Textbook"
        ]
    },
    6: {
        "name": "Technology & Applied Sciences",
        "name_ko": "기술, 응용과학",
        "color": "#b3de69",  # 연두
        "keywords": [
            "technology", "engineering", "medicine", "health", "agriculture",
            "cooking", "manufacturing", "construction", "electronics",
            "mechanical", "chemical engineering", "biotechnology",
            "nutrition", "diet", "fitness", "disease", "surgery", "nursing",
            "architecture", "design", "invention", "innovation"
        ],
        "examples": [
            "The Design of Everyday Things",
            "How to Cook Everything",
            "Gray's Anatomy",
            "Engineering Mechanics",
            "The Innovator's Dilemma",
            "Introduction to Robotics"
        ]
    },
    7: {
        "name": "Arts & Recreation",
        "name_ko": "예술, 오락",
        "color": "#fccde5",  # 연분홍
        "keywords": [
            "art", "music", "painting", "sculpture", "photography", "film",
            "theater", "dance", "architecture", "design", "fashion",
            "sports", "games", "hobbies", "crafts", "gardening",
            "entertainment", "performance", "aesthetic", "beauty",
            "drawing", "illustration", "animation", "cinema"
        ],
        "examples": [
            "The Story of Art by E.H. Gombrich",
            "Ways of Seeing by John Berger",
            "Understanding Comics by Scott McCloud",
            "The Art Spirit",
            "A History of Western Music",
            "The Game: Penetrating the Secret Society of Pickup Artists"
        ]
    },
    8: {
        "name": "Literature",
        "name_ko": "문학",
        "color": "#d9d9d9",  # 연회색
        "keywords": [
            "fiction", "novel", "poetry", "drama", "short story", "essay",
            "literary", "narrative", "prose", "verse", "author", "writer",
            "classic", "contemporary", "romance", "mystery", "thriller",
            "fantasy", "science fiction", "horror", "adventure",
            "literary criticism", "rhetoric", "creative writing"
        ],
        "examples": [
            "Pride and Prejudice by Jane Austen",
            "1984 by George Orwell",
            "To Kill a Mockingbird",
            "The Great Gatsby",
            "Harry Potter Series",
            "The Lord of the Rings"
        ]
    },
    9: {
        "name": "History & Geography",
        "name_ko": "역사, 지리",
        "color": "#bc80bd",  # 연자주
        "keywords": [
            "history", "geography", "biography", "civilization", "war",
            "ancient", "medieval", "modern", "century", "empire", "kingdom",
            "revolution", "world war", "colonialism", "archaeology",
            "travel", "exploration", "maps", "countries", "continents",
            "culture", "heritage", "tradition", "memoir", "autobiography"
        ],
        "examples": [
            "Guns, Germs, and Steel by Jared Diamond",
            "A People's History of the United States",
            "The Diary of a Young Girl by Anne Frank",
            "The Rise and Fall of the Third Reich",
            "Longitude: The True Story",
            "Into the Wild by Jon Krakauer"
        ]
    }
}

# Unknown 카테고리
DDC_UNKNOWN = 10


def parse_args():
    p = argparse.ArgumentParser(description="Assign DDC classes using embedding similarity")
    p.add_argument("--snippet_json", required=True, help="Path to id2snippet JSON file")
    p.add_argument("--ids_file", required=True, help="Path to ids .uint32 file")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--tag", default="", help="Output filename tag")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--top_k", type=int, default=1, help="Use top-k similar DDC classes (1=best only)")
    p.add_argument("--confidence_threshold", type=float, default=0.0, 
                   help="Minimum similarity score, below this → unknown (10)")
    return p.parse_args()


def build_ddc_texts():
    """
    DDC별 대표 텍스트 생성
    키워드 + 예시를 조합하여 각 DDC를 대표하는 텍스트 만들기
    """
    ddc_texts = {}
    for ddc_id, info in DDC_CLASSES.items():
        # 키워드와 예시를 조합
        keywords_text = ", ".join(info["keywords"])
        examples_text = ". ".join(info["examples"])
        
        # 여러 버전의 대표 텍스트 생성 (다양성을 위해)
        texts = [
            f"{info['name']}. Topics include: {keywords_text}",
            f"Books about {keywords_text}",
            examples_text,
        ]
        ddc_texts[ddc_id] = texts
    
    return ddc_texts


def main():
    args = parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 출력 파일 경로
    tag = f"_{args.tag}" if args.tag else ""
    ddc_out_path = out_dir / f"ddc{tag}.uint16"
    ddc_meta_path = out_dir / f"ddc_meta{tag}.json"
    
    print(f"Loading snippet JSON: {args.snippet_json}")
    with open(args.snippet_json, "r", encoding="utf-8") as f:
        id2snippet = json.load(f)
    
    print(f"Loading IDs: {args.ids_file}")
    ids = np.fromfile(args.ids_file, dtype=np.uint32)
    n = len(ids)
    print(f"  → {n} points")
    
    # 각 ID에 대한 텍스트 추출
    texts = []
    valid_mask = []
    for book_id in ids:
        snippet = id2snippet.get(str(book_id), "")
        texts.append(snippet if snippet else "")
        valid_mask.append(bool(snippet))
    
    print(f"  → {sum(valid_mask)} / {n} have valid text")
    
    # 모델 로드
    print(f"\nLoading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    
    # DDC 대표 텍스트 임베딩
    print("\nEmbedding DDC representative texts...")
    ddc_texts = build_ddc_texts()
    
    ddc_embeddings = {}  # ddc_id -> averaged embedding
    for ddc_id, text_list in tqdm(ddc_texts.items(), desc="DDC classes"):
        embs = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)
        # 평균 임베딩
        ddc_embeddings[ddc_id] = embs.mean(axis=0)
    
    # DDC 임베딩을 행렬로 변환 (10 x dim)
    ddc_ids_ordered = sorted(ddc_embeddings.keys())
    ddc_matrix = np.stack([ddc_embeddings[i] for i in ddc_ids_ordered])  # (10, dim)
    
    # 책 텍스트 임베딩
    print(f"\nEmbedding {n} book texts...")
    book_embeddings = model.encode(
        texts, 
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )  # (n, dim)
    
    # 코사인 유사도 계산 (정규화된 벡터이므로 내적 = 코사인 유사도)
    print("\nComputing similarities...")
    similarities = book_embeddings @ ddc_matrix.T  # (n, 10)
    
    # 가장 유사한 DDC 할당
    best_ddc = similarities.argmax(axis=1)  # (n,)
    best_scores = similarities.max(axis=1)  # (n,)
    
    # 신뢰도 낮으면 unknown
    ddc_assignments = np.where(
        best_scores >= args.confidence_threshold,
        best_ddc,
        DDC_UNKNOWN
    ).astype(np.uint16)
    
    # 통계 출력
    print("\n" + "=" * 50)
    print("DDC Assignment Statistics")
    print("=" * 50)
    
    for ddc_id in range(11):
        count = (ddc_assignments == ddc_id).sum()
        pct = count / n * 100
        if ddc_id < 10:
            name = DDC_CLASSES[ddc_id]["name_ko"]
        else:
            name = "Unknown"
        print(f"  {ddc_id}: {name:20s} | {count:5d} ({pct:5.1f}%)")
    
    # 점수 분포
    print(f"\nSimilarity scores: min={best_scores.min():.3f}, max={best_scores.max():.3f}, mean={best_scores.mean():.3f}")
    
    # 저장
    print(f"\nSaving DDC assignments to: {ddc_out_path}")
    ddc_assignments.tofile(ddc_out_path)
    
    # 메타데이터 저장
    meta = {
        "n": n,
        "model": args.model,
        "num_classes": 11,
        "classes": {
            str(k): {
                "name": v["name"],
                "name_ko": v["name_ko"],
                "color": v["color"]
            } for k, v in DDC_CLASSES.items()
        },
        "unknown_class": DDC_UNKNOWN,
        "files": {
            "ddc": str(ddc_out_path)
        }
    }
    meta["classes"]["10"] = {
        "name": "Unknown",
        "name_ko": "미분류",
        "color": "#969696"
    }
    
    print(f"Saving metadata to: {ddc_meta_path}")
    with open(ddc_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print("\nDone!")


if __name__ == "__main__":
    main()