"""
data_prep_coursera.py
=====================
Loads the Coursera Courses Dataset 2021 catalog and prepares it for Task 2 BERT.

Input :  data/raw/coursera/Coursera.csv  (3,522 courses, full descriptions)
Output:  data/processed/coursera_clean.csv

Steps:
  1. Drop rows with missing description / name.
  2. Clean text: collapse whitespace, fix mojibake, keep alphanumeric punctuation.
  3. Build a single 'text' field combining name + description + skills (BERT input).
  4. Bucket courses by topic from skills/text -- used for plot colour and aligning
     with OULAD domains in the comparison section.
  5. Down-select to a tractable subset (configurable) -- by default we keep
     all 3,522 courses, but the BERT script will sample if torch is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "coursera"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


# Topic buckets aligned with OULAD's STEM/SocSci domains so we can compare BERT-similarity
# with PageRank/FP-Growth at the domain level.
TOPIC_RULES = [
    ("Computing & IT",     ["python", "programming", "javascript", "software", "web develop",
                            "data structure", "algorithm", "computer science", "coding",
                            "linux", "shell", "database", "sql", "cloud", "aws", "devops"]),
    ("Mathematics",        ["calculus", "algebra", "probability", "statistics", "discrete math",
                            "linear algebra", "stochastic", "mathematical"]),
    ("Engineering",        ["engineering", "robotics", "control systems", "electrical",
                            "mechanical", "civil ", "manufacturing"]),
    ("Technology Studies", ["machine learning", "deep learning", "ai ", "artificial intel",
                            "data science", "neural", "tensorflow", "pytorch", "nlp",
                            "computer vision", "iot ", "blockchain"]),
    ("Humanities",         ["history", "philosophy", "literature", "writing", "art ",
                            "music", "language", "creative", "screenplay"]),
    ("Education Studies",  ["teaching", "education", "pedagog", "learning theory",
                            "curriculum", "classroom"]),
    ("Social Sciences Foundation",
                           ["sociology", "psychology", "anthropology", "social science",
                            "behaviour", "behavior", "research methods"]),
]


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Replace common mojibake artefacts
    s = (s.replace("’", "'").replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-")
          .replace("�", " ").replace("\xa0", " "))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def assign_topic(text: str) -> str:
    text_lc = text.lower()
    for label, keywords in TOPIC_RULES:
        for kw in keywords:
            if kw in text_lc:
                return label
    return "Other"


def main() -> None:
    print("[coursera] Loading Coursera.csv")
    df = pd.read_csv(RAW / "Coursera.csv", encoding="latin-1")
    print(f"[coursera] Raw shape: {df.shape}")

    df = df.dropna(subset=["Course Name", "Course Description"]).copy()
    df["course_name"] = df["Course Name"].apply(clean_text)
    df["description"] = df["Course Description"].apply(clean_text)
    df["skills"] = df["Skills"].fillna("").apply(clean_text)
    df["university"] = df["University"].fillna("").apply(clean_text)
    df["difficulty"] = df["Difficulty Level"].fillna("Unknown")
    df["rating"] = pd.to_numeric(df["Course Rating"], errors="coerce")
    df["url"] = df["Course URL"].fillna("")

    # Build a single text blob for BERT
    df["bert_text"] = (df["course_name"] + ". "
                       + df["description"] + " "
                       + df["skills"]).str.slice(0, 2000)

    # Topic bucketing
    combined = (df["course_name"] + " " + df["description"] + " " + df["skills"]).str.lower()
    df["topic"] = combined.apply(assign_topic)

    keep = ["course_name", "university", "difficulty", "rating", "url",
            "description", "skills", "bert_text", "topic"]
    out = df[keep].drop_duplicates(subset="course_name").reset_index(drop=True)
    out["course_id"] = "CRS" + (out.index + 1).astype(str).str.zfill(5)
    out = out[["course_id"] + keep]

    out.to_csv(PROCESSED / "coursera_clean.csv", index=False)
    print(f"[coursera] Cleaned shape: {out.shape}")
    print("\n[coursera] Topic distribution:")
    print(out["topic"].value_counts().to_string())
    print("\n[coursera] Difficulty distribution:")
    print(out["difficulty"].value_counts().to_string())
    print("\n[coursera] Description length stats (chars):")
    print(out["description"].str.len().describe().round(0).to_string())
    print("\n[coursera] Top 3 sample rows:")
    print(out[["course_id", "course_name", "topic", "difficulty"]].head(3).to_string(index=False))
    print("[coursera] Done.")


if __name__ == "__main__":
    main()
