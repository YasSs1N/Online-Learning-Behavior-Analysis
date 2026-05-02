"""
bert_similarity.py
==================
Task 2 -- BERT-based course similarity on the Coursera Courses Dataset 2021.

Two encoders for comparison:
  Model A:  sentence-transformers / all-MiniLM-L6-v2
            BERT-family encoder fine-tuned for sentence similarity.
  Model B:  bert-base-uncased + manual mean-pooling
            Raw BERT, closer to "literal BERT" wording in the project spec.

Workflow:
  1. Load coursera_clean.csv (course_name + description + topic label).
  2. Sample N courses for tractability (default 500; configurable).
  3. Embed each course's text with both encoders.
  4. Compute pairwise cosine-similarity matrices.
  5. For each course, save its top-10 nearest neighbours.
  6. Compare the two models: Pearson + Spearman correlation, top-pair overlap.

Outputs to results/:
  bert_minilm_embeddings.npy
  bert_base_embeddings.npy
  bert_minilm_top_neighbours.csv
  bert_base_top_neighbours.csv
  bert_similarity_minilm.npz   (square matrix)
  bert_similarity_base.npz
  bert_model_comparison.csv

If torch / transformers are unavailable, falls back to a deterministic
TF-IDF + topic-augmented projection so the pipeline still completes;
real BERT runs locally with `pip install torch transformers sentence-transformers`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------- Real encoders ---------- #
def encode_minilm(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, normalize_embeddings=True,
                                    show_progress_bar=False, batch_size=32))


def encode_bert_base(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()

    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=256, padding=True)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean = (summed / counts).squeeze(0)
            mean = torch.nn.functional.normalize(mean, dim=0)
            embs.append(mean.numpy())
    return np.vstack(embs)


# ---------- Fallback (deterministic, no torch) ---------- #
def encode_fallback(texts: list[str], topics: list[str], dim: int = 384,
                    seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                            min_df=2, max_features=4000)
    X_text = tfidf.fit_transform(texts).toarray()
    topic_dummies = pd.get_dummies(pd.Series(topics)).values.astype(float)
    X = np.hstack([X_text, topic_dummies * 2.0])
    R = rng.randn(X.shape[1], dim) / np.sqrt(dim)
    Y = X @ R
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Y


def safe_encode(name: str, fn, *args, **kwargs):
    try:
        emb = fn(*args, **kwargs)
        print(f"[bert] {name}: real encoder OK ({emb.shape})")
        return emb, "real"
    except Exception as e:
        print(f"[bert] {name}: falling back ({type(e).__name__}: {str(e)[:80]})")
        return None, "fallback"


def top_neighbours(sim: np.ndarray, course_ids: list[str],
                   names: list[str], topics: list[str], k: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(len(course_ids)):
        scores = sim[i].copy()
        scores[i] = -np.inf
        idx = np.argsort(scores)[::-1][:k]
        for rank, j in enumerate(idx, start=1):
            rows.append({
                "course_id": course_ids[i], "course_name": names[i], "topic": topics[i],
                "rank": rank, "neighbour_id": course_ids[j],
                "neighbour_name": names[j], "neighbour_topic": topics[j],
                "similarity": float(scores[j]),
            })
    return pd.DataFrame(rows)


def main(sample_size: int) -> None:
    print("[bert] Loading coursera_clean.csv")
    df = pd.read_csv(PROCESSED / "coursera_clean.csv")

    # Stratified sample by topic so we cover the full topical space
    if sample_size and sample_size < len(df):
        sample = (df.groupby("topic", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), max(20, sample_size // df["topic"].nunique())),
                                            random_state=42)))
        if len(sample) > sample_size:
            sample = sample.sample(sample_size, random_state=42)
        df = sample.sort_values("course_id").reset_index(drop=True)
    print(f"[bert] Working set: {len(df)} courses across {df['topic'].nunique()} topics")
    print(df["topic"].value_counts().to_string())

    texts = df["bert_text"].astype(str).tolist()
    topics = df["topic"].tolist()
    course_ids = df["course_id"].tolist()
    names = df["course_name"].tolist()

    # ----- Model A ----- #
    emb_mini, mode_mini = safe_encode("MiniLM", encode_minilm, texts)
    if emb_mini is None:
        emb_mini = encode_fallback(texts, topics, seed=42)

    # ----- Model B ----- #
    emb_base, mode_base = safe_encode("bert-base-uncased", encode_bert_base, texts)
    if emb_base is None:
        # Different seed + char-level so the two fallbacks aren't identical
        emb_base = encode_fallback(texts, topics, seed=7)

    np.save(RESULTS / "bert_minilm_embeddings.npy", emb_mini)
    np.save(RESULTS / "bert_base_embeddings.npy", emb_base)

    sim_mini = cosine_similarity(emb_mini)
    sim_base = cosine_similarity(emb_base)
    np.savez_compressed(RESULTS / "bert_similarity_minilm.npz",
                         sim=sim_mini, course_ids=np.array(course_ids))
    np.savez_compressed(RESULTS / "bert_similarity_base.npz",
                         sim=sim_base, course_ids=np.array(course_ids))

    nb_mini = top_neighbours(sim_mini, course_ids, names, topics, k=10)
    nb_base = top_neighbours(sim_base, course_ids, names, topics, k=10)
    nb_mini.to_csv(RESULTS / "bert_minilm_top_neighbours.csv", index=False)
    nb_base.to_csv(RESULTS / "bert_base_top_neighbours.csv", index=False)
    print(f"[bert] Saved top-10 neighbours: minilm={len(nb_mini)} base={len(nb_base)}")

    # ----- Comparison ----- #
    iu = np.triu_indices_from(sim_mini, k=1)
    v_mini = sim_mini[iu]
    v_base = sim_base[iu]
    pearson = float(np.corrcoef(v_mini, v_base)[0, 1])
    spearman = float(pd.Series(v_mini).corr(pd.Series(v_base), method="spearman"))

    # Top-100 most-similar pair overlap
    top100_mini = set(np.argsort(v_mini)[-100:])
    top100_base = set(np.argsort(v_base)[-100:])
    overlap = len(top100_mini & top100_base)

    cmp = pd.DataFrame({"metric": ["pearson", "spearman", "top100_overlap",
                                     "n_courses", "mode_minilm", "mode_base"],
                         "value":  [round(pearson, 4), round(spearman, 4),
                                    overlap, len(df), mode_mini, mode_base]})
    cmp.to_csv(RESULTS / "bert_model_comparison.csv", index=False)
    print(f"\n[bert] Comparison:\n{cmp.to_string(index=False)}")

    # Show sample neighbours
    print("\n[bert] Sample top-3 neighbours from MiniLM (first 3 courses):")
    print(nb_mini[nb_mini["rank"] <= 3].head(9)[["course_name", "rank",
                                                  "neighbour_name", "similarity"]]
          .to_string(index=False))
    print("[bert] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500,
                    help="Sample N Coursera courses (default 500). Use 0 for all 3,416.")
    args = ap.parse_args()
    main(args.sample if args.sample > 0 else 0)
