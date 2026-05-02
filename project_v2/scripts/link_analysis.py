"""
link_analysis.py
================
PageRank + HITS on the OULAD course co-enrolment graph.

Input :  data/processed/course_graph_edges.csv  (source, target, weight)
         data/processed/course_meta.csv
Output:  results/course_pagerank.csv
         results/course_hits.csv
         results/course_graph.graphml
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    edges = pd.read_csv(PROCESSED / "course_graph_edges.csv")
    meta = pd.read_csv(PROCESSED / "course_meta.csv")

    print(f"[link] Edges: {len(edges)}, Modules: {len(meta)}")

    # Build undirected weighted graph: nodes are modules, edges are co-enrolment counts
    G = nx.Graph()
    for _, m in meta.iterrows():
        G.add_node(m["module_name"], code=m["code_module"], domain=m["domain"],
                   subject_area=m["subject_area"], students=int(m["students"]))
    for _, e in edges.iterrows():
        G.add_edge(e["source"], e["target"], weight=int(e["weight"]))
    nx.write_graphml(G, RESULTS / "course_graph.graphml")
    print(f"[link] Graph: nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

    # Weighted PageRank (alpha=0.85)
    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    pr_df = pd.DataFrame({"module_name": list(pr.keys()),
                          "pagerank_score": list(pr.values())})
    pr_df = pr_df.merge(meta[["code_module", "module_name", "domain",
                              "subject_area", "students"]], on="module_name")
    pr_df = pr_df.sort_values("pagerank_score", ascending=False).reset_index(drop=True)
    pr_df["rank"] = pr_df.index + 1
    pr_df.to_csv(RESULTS / "course_pagerank.csv", index=False)
    print("\n[link] PageRank ranking:")
    print(pr_df[["rank", "module_name", "domain", "students", "pagerank_score"]]
          .to_string(index=False))

    # HITS (hubs and authorities) -- on the same graph
    h, a = nx.hits(G, max_iter=200, tol=1e-8)
    hits_df = pd.DataFrame({"module_name": list(h.keys()),
                            "hub_score": list(h.values()),
                            "authority_score": [a[k] for k in h.keys()]})
    hits_df = hits_df.merge(meta[["module_name", "domain"]], on="module_name")
    hits_df = hits_df.sort_values("authority_score", ascending=False).reset_index(drop=True)
    hits_df.to_csv(RESULTS / "course_hits.csv", index=False)
    print("\n[link] HITS (top by authority):")
    print(hits_df[["module_name", "domain", "hub_score", "authority_score"]]
          .to_string(index=False))
    print("[link] Done.")


if __name__ == "__main__":
    main()
