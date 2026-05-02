"""
visualization.py
================
Generates all figures for the project. Uses readable module names (from the
mapping Excel), never raw codes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

DOMAIN_COLORS = {"STEM": "#2C7FB8", "Social Sciences": "#E66101"}


def save(fig, name: str) -> None:
    out = FIGS / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] {out.name}")


def fig_oulad_overview() -> None:
    meta = pd.read_csv(PROCESSED / "course_meta.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    meta_sorted = meta.sort_values("students", ascending=False)
    colors = [DOMAIN_COLORS[d] for d in meta_sorted["domain"]]
    axes[0, 0].bar(meta_sorted["module_name"], meta_sorted["students"], color=colors)
    axes[0, 0].set_title("Students Enrolled per Module")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("Students")

    meta_pass = meta.sort_values("pct_pass", ascending=False)
    colors = [DOMAIN_COLORS[d] for d in meta_pass["domain"]]
    axes[0, 1].bar(meta_pass["module_name"], meta_pass["pct_pass"], color=colors)
    axes[0, 1].set_title("Pass Rate per Module")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set_ylabel("Pass rate")

    outcome_cols = ["pct_pass", "pct_distinction", "pct_fail", "pct_withdrawn"]
    outcomes = meta.set_index("module_name")[outcome_cols].rename(columns={
        "pct_pass": "Pass", "pct_distinction": "Distinction",
        "pct_fail": "Fail", "pct_withdrawn": "Withdrawn"})
    outcomes.plot(kind="bar", stacked=True, ax=axes[1, 0],
                  color=["#67A9CF", "#1A9850", "#D7191C", "#999999"])
    axes[1, 0].set_title("Final-Result Mix per Module")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")

    txn = pd.read_csv(PROCESSED / "transactions.csv")
    counts = txn["n_modules"].value_counts().sort_index()
    axes[1, 1].bar(counts.index.astype(str), counts.values, color="steelblue")
    axes[1, 1].set_title("Modules per Student (multi-enrolments)")
    axes[1, 1].set_xlabel("Modules")
    axes[1, 1].set_ylabel("Students (log scale)")
    axes[1, 1].set_yscale("log")
    for x, y in zip(counts.index.astype(str), counts.values):
        axes[1, 1].text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("OULAD overview — readable module names", fontsize=14, y=1.02)
    save(fig, "01_oulad_overview")


def fig_association_rules() -> None:
    rules = pd.read_csv(RESULTS / "association_rules.csv")
    if rules.empty:
        print("[viz] No rules to plot")
        return
    rules["rule"] = rules["antecedents_str"] + "  →  " + rules["consequents_str"]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(rules) + 2)))
    sns.barplot(data=rules.sort_values("lift"),
                y="rule", x="lift", color="#1F3A68", ax=ax)
    for i, (_, r) in enumerate(rules.sort_values("lift").iterrows()):
        ax.text(r["lift"] + 0.02, i,
                f"sup={r['support']:.3f}  conf={r['confidence']:.2f}",
                va="center", fontsize=9)
    ax.set_title("Course-level association rules (FP-Growth, sorted by lift)")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")
    save(fig, "02_association_rules")


def fig_course_graph() -> None:
    pr = pd.read_csv(RESULTS / "course_pagerank.csv").set_index("module_name")
    G = nx.read_graphml(RESULTS / "course_graph.graphml")
    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.6, iterations=200)

    fig, ax = plt.subplots(figsize=(11, 8))
    sizes = [3500 * pr.loc[n, "pagerank_score"] + 400 for n in G.nodes()]
    colors = [DOMAIN_COLORS[G.nodes[n].get("domain", "STEM")] for n in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.3 + 6 * (w / max_w) for w in weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="#888888", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                            edgecolors="black", linewidths=1.5, alpha=0.92, ax=ax)
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10,
                             font_weight="bold", ax=ax)
    edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=8, ax=ax)
    ax.set_title("Course co-enrolment network (node size = PageRank)")
    ax.axis("off")

    handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=c,
                          markersize=12, label=d, markeredgecolor='black')
               for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", title="Domain")
    save(fig, "03_course_graph_pagerank")


def fig_pagerank_bar() -> None:
    pr = pd.read_csv(RESULTS / "course_pagerank.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    pr_s = pr.sort_values("pagerank_score")
    colors = [DOMAIN_COLORS[d] for d in pr_s["domain"]]
    ax.barh(pr_s["module_name"], pr_s["pagerank_score"], color=colors)
    for y, (v, n) in enumerate(zip(pr_s["pagerank_score"], pr_s["students"])):
        ax.text(v + 0.005, y, f"{v:.3f}  (n={n:,})", va="center", fontsize=9)
    ax.set_title("PageRank ranking of OULAD modules")
    ax.set_xlabel("PageRank score")
    ax.set_xlim(0, max(pr["pagerank_score"]) * 1.25)
    handles = [plt.Line2D([], [], marker='s', color='w', markerfacecolor=c,
                          markersize=12, label=d) for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", title="Domain")
    save(fig, "04_pagerank_bar")


def fig_hits() -> None:
    hits = pd.read_csv(RESULTS / "course_hits.csv").sort_values("authority_score")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(hits))
    w = 0.4
    ax.barh(x - w/2, hits["hub_score"], w, label="Hub", color="#1F77B4")
    ax.barh(x + w/2, hits["authority_score"], w, label="Authority", color="#FF7F0E")
    ax.set_yticks(x)
    ax.set_yticklabels(hits["module_name"])
    ax.set_xlabel("Score")
    ax.set_title("HITS hub & authority scores per module")
    ax.legend()
    save(fig, "05_hits")


def fig_bert_topics() -> None:
    """BERT similarity heatmap aggregated to TOPIC level (so 8x8, readable)."""
    nb = pd.read_csv(RESULTS / "bert_minilm_top_neighbours.csv")

    # Average similarity within & between topics
    pivot = (nb.groupby(["topic", "neighbour_topic"])["similarity"]
             .mean().unstack(fill_value=np.nan))
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Mean cosine similarity"}, ax=ax)
    ax.set_title("BERT similarity aggregated by topic (MiniLM)")
    ax.set_xlabel("Neighbour topic")
    ax.set_ylabel("Source topic")
    save(fig, "06_bert_topic_heatmap")


def fig_bert_compare() -> None:
    cmp = pd.read_csv(RESULTS / "bert_model_comparison.csv")
    nb_m = pd.read_csv(RESULTS / "bert_minilm_top_neighbours.csv")
    nb_b = pd.read_csv(RESULTS / "bert_base_top_neighbours.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of top-1 similarities
    top1_m = nb_m[nb_m["rank"] == 1]["similarity"]
    top1_b = nb_b[nb_b["rank"] == 1]["similarity"]
    axes[0].hist([top1_m, top1_b], bins=30, label=["MiniLM", "bert-base-uncased"],
                  color=["#5B8DEF", "#F39C12"], alpha=0.85)
    axes[0].set_title("Top-1 nearest-neighbour similarity distribution")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].legend()

    # Per-course top-1 agreement: fraction of courses whose top-1 neighbour matches
    pivot_m = nb_m[nb_m["rank"] == 1].set_index("course_id")["neighbour_id"]
    pivot_b = nb_b[nb_b["rank"] == 1].set_index("course_id")["neighbour_id"]
    common = pivot_m.index.intersection(pivot_b.index)
    agree_at1 = (pivot_m.loc[common] == pivot_b.loc[common]).mean()

    axes[1].axis("off")
    axes[1].set_title("Model comparison summary")
    txt = (f"Pearson(sim_minilm, sim_base) = {float(cmp[cmp['metric']=='pearson']['value'].iloc[0]):.3f}\n"
           f"Spearman                       = {float(cmp[cmp['metric']=='spearman']['value'].iloc[0]):.3f}\n"
           f"Top-100 pair overlap          = {int(cmp[cmp['metric']=='top100_overlap']['value'].iloc[0])} / 100\n"
           f"Top-1 neighbour agreement   = {agree_at1*100:.1f}%\n\n"
           f"Courses analysed              = {int(cmp[cmp['metric']=='n_courses']['value'].iloc[0]):,}\n"
           f"MiniLM mode                   = {cmp[cmp['metric']=='mode_minilm']['value'].iloc[0]}\n"
           f"bert-base-uncased mode    = {cmp[cmp['metric']=='mode_base']['value'].iloc[0]}")
    axes[1].text(0.05, 0.95, txt, family="monospace", fontsize=11,
                  va="top", transform=axes[1].transAxes)

    fig.suptitle("Task 2 — BERT model comparison", fontsize=13, y=1.02)
    save(fig, "07_bert_model_comparison")


def fig_neighbour_examples() -> None:
    nb = pd.read_csv(RESULTS / "bert_minilm_top_neighbours.csv")
    examples = (nb[nb["rank"] <= 5]
                .groupby("topic", group_keys=False)
                .apply(lambda g: g.head(15)).head(40))
    fig, ax = plt.subplots(figsize=(13, 8))
    pivot = examples.pivot_table(index="course_name",
                                  columns="rank",
                                  values="similarity",
                                  aggfunc="first").fillna(0)
    pivot = pivot.iloc[:20]
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Top-5 BERT neighbours' similarity (first 20 sample courses)")
    ax.set_xlabel("Neighbour rank")
    ax.set_ylabel("")
    save(fig, "08_bert_top5_examples")


def main() -> None:
    fig_oulad_overview()
    fig_association_rules()
    fig_course_graph()
    fig_pagerank_bar()
    fig_hits()
    fig_bert_topics()
    fig_bert_compare()
    fig_neighbour_examples()
    print("[viz] Done.")


if __name__ == "__main__":
    main()
