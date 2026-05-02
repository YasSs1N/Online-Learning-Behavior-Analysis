"""
learning_paths.py
=================
Combines FP-Growth course rules + PageRank + BERT similarity into a single
recommended-paths table.

Outputs:
  results/recommended_paths.csv

Logic:
  - For each OULAD module (sorted by PageRank), find association-rule successors:
      antecedent = {module} -> consequent = {next_module}
  - For each Coursera topic, surface the most BERT-similar course pairs as
    "semantic-neighbour pairs" -- a recommendation row.
  - Combine into a single dataframe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    pr = pd.read_csv(RESULTS / "course_pagerank.csv")
    rules = pd.read_csv(RESULTS / "association_rules.csv")
    bert_nb = pd.read_csv(RESULTS / "bert_minilm_top_neighbours.csv")

    rows = []

    # ---- Pattern-driven OULAD recommendations ---- #
    for _, p in pr.iterrows():
        succ = rules[(rules["antecedents_str"] == p["module_name"])
                     & (rules["consequents_str"] != p["module_name"])]
        succ = succ.sort_values("lift", ascending=False).head(3)
        for _, r in succ.iterrows():
            rows.append({
                "source_dataset": "OULAD",
                "from_course": p["module_name"],
                "to_course": r["consequents_str"],
                "evidence": "FP-Growth",
                "support": round(r["support"], 4),
                "confidence": round(r["confidence"], 3),
                "lift": round(r["lift"], 3),
                "pagerank_source": round(p["pagerank_score"], 4),
                "rationale": (f"Students who took '{p['module_name']}' also took "
                              f"'{r['consequents_str']}' "
                              f"(lift={r['lift']:.2f}, conf={r['confidence']:.2f})."),
            })

    # ---- BERT-driven Coursera recommendations ---- #
    # For each topic, take the highest-similarity top-1 neighbour for each of
    # the top 3 courses (by within-topic representativeness)
    top_by_topic = (bert_nb[bert_nb["rank"] == 1]
                    .sort_values("similarity", ascending=False)
                    .groupby("topic", group_keys=False).head(3))
    for _, b in top_by_topic.iterrows():
        rows.append({
            "source_dataset": "Coursera",
            "from_course": b["course_name"],
            "to_course": b["neighbour_name"],
            "evidence": "BERT (MiniLM)",
            "support": pd.NA,
            "confidence": pd.NA,
            "lift": pd.NA,
            "pagerank_source": pd.NA,
            "rationale": (f"Most semantically similar course "
                          f"(cosine={b['similarity']:.3f}; topic={b['topic']})."),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "recommended_paths.csv", index=False)
    print(f"[paths] Wrote {len(out)} recommendations")
    print("\n[paths] Sample (OULAD pattern-driven):")
    print(out[out["source_dataset"] == "OULAD"]
          [["from_course", "to_course", "lift", "confidence"]].to_string(index=False))
    print(f"\n[paths] Sample (Coursera BERT, first 12):")
    print(out[out["source_dataset"] == "Coursera"]
          [["from_course", "to_course"]].head(12).to_string(index=False))
    print("[paths] Done.")


if __name__ == "__main__":
    main()
