import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
enrollments = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/uploads/Cleaned_Enrollments.csv')
print("Enrollment data shape:", enrollments.shape)
print("Columns:", enrollments.columns.tolist())
print("Unique courses:", enrollments['course_name'].nunique())
print("Unique users:", enrollments['userid_DI'].nunique())

# ─── BUILD TRANSACTION MATRIX ────────────────────────────────────────────────
# Group by user -> list of courses (transactions)
transactions = enrollments.groupby('userid_DI')['course_name'].apply(list).tolist()
print(f"\nTotal transactions (users): {len(transactions)}")
print(f"Sample transaction: {transactions[0]}")

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)
print(f"\nTransaction matrix: {df_te.shape}")

# ─── TASK 1A: APRIORI ─────────────────────────────────────────────────────────
print("\n=== APRIORI ===")
frequent_itemsets = apriori(df_te, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False).reset_index(drop=True)
frequent_itemsets['count'] = (frequent_itemsets['support'] * len(df_te)).round().astype(int)
frequent_itemsets.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/frequent_itemsets.csv', index=False)
print(f"Frequent itemsets: {len(frequent_itemsets)}")

# Association rules from Apriori
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.05)
rules['antecedents'] = rules['antecedents'].apply(lambda x: str(x))
rules['consequents'] = rules['consequents'].apply(lambda x: str(x))
rules['certainty_factor'] = (rules['confidence'] - rules['support']) / (1 - rules['support'])
rules['count'] = (rules['support'] * len(df_te)).round().astype(int)
rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
rules.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/association_rules.csv', index=False)
print(f"Association rules: {len(rules)}")
print(rules[['antecedents','consequents','support','confidence','lift','leverage','conviction','certainty_factor','count']].head(5))

# ─── TASK 1B: FP-GROWTH ──────────────────────────────────────────────────────
print("\n=== FP-GROWTH ===")
fp_frequent = fpgrowth(df_te, min_support=0.01, use_colnames=True)
fp_frequent['length'] = fp_frequent['itemsets'].apply(len)
fp_frequent = fp_frequent.sort_values('support', ascending=False).reset_index(drop=True)
fp_frequent['count'] = (fp_frequent['support'] * len(df_te)).round().astype(int)
fp_frequent.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/fp_frequent_itemsets.csv', index=False)
print(f"FP-Growth frequent itemsets: {len(fp_frequent)}")

fp_rules = association_rules(fp_frequent, metric='confidence', min_threshold=0.05)
fp_rules['antecedents'] = fp_rules['antecedents'].apply(lambda x: str(x))
fp_rules['consequents'] = fp_rules['consequents'].apply(lambda x: str(x))
fp_rules['certainty_factor'] = (fp_rules['confidence'] - fp_rules['support']) / (1 - fp_rules['support'])
fp_rules['count'] = (fp_rules['support'] * len(df_te)).round().astype(int)
fp_rules = fp_rules.sort_values('lift', ascending=False).reset_index(drop=True)
fp_rules.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/fp_rules.csv', index=False)
print(f"FP-Growth rules: {len(fp_rules)}")

# ─── TASK 1C: PAGERANK / LINK ANALYSIS ──────────────────────────────────────
print("\n=== PAGERANK ===")
# Build course co-enrollment graph: edge between 2 courses if taken by same user
G = nx.Graph()
all_courses = enrollments['course_name'].unique().tolist()
G.add_nodes_from(all_courses)

# Build edges from user transactions
edge_weights = {}
for tx in transactions:
    unique_courses = list(set(tx))
    for i in range(len(unique_courses)):
        for j in range(i+1, len(unique_courses)):
            pair = tuple(sorted([unique_courses[i], unique_courses[j]]))
            edge_weights[pair] = edge_weights.get(pair, 0) + 1

for (c1, c2), w in edge_weights.items():
    G.add_edge(c1, c2, weight=w)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save edges
edges_df = pd.DataFrame(
    [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
    columns=['source', 'target', 'weight']
).sort_values('weight', ascending=False)
edges_df.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/course_graph_edges.csv', index=False)

# PageRank
pagerank = nx.pagerank(G, weight='weight')
pr_df = pd.DataFrame({'course_name': list(pagerank.keys()), 'pagerank_score': list(pagerank.values())})
pr_df = pr_df.sort_values('pagerank_score', ascending=False).reset_index(drop=True)
pr_df['rank'] = pr_df.index + 1
pr_df.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/pagerank_scores.csv', index=False)
print("\nTop PageRank courses:")
print(pr_df.head(10))

print("\n✅ Task 1 complete — all CSVs saved.")
