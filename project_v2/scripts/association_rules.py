"""
association_rules.py
====================
Course-level FP-Growth + Apriori on real OULAD multi-enrolments.

Input :  data/processed/transactions.csv  (per-student JSON list of module_names)
Output:  results/frequent_itemsets_fpgrowth.csv
         results/frequent_itemsets_apriori.csv
         results/association_rules.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# Tuned for OULAD's small co-enrolment volume (~2,500 students with >=2 modules)
MIN_SUPPORT = 0.005    # 0.5% of all students
MIN_CONFIDENCE = 0.10
MIN_LIFT = 1.0


def fmt_set(s) -> str:
    return ", ".join(sorted(s))


def main() -> None:
    print("[rules] Loading transactions")
    txn = pd.read_csv(PROCESSED / "transactions.csv")
    txn["modules"] = txn["modules"].apply(json.loads)

    baskets = txn["modules"].tolist()
    print(f"[rules] Total baskets: {len(baskets):,}")
    print(f"[rules] Baskets with >=2 modules: {sum(1 for b in baskets if len(b) >= 2):,}")

    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    encoded = pd.DataFrame(te_array, columns=te.columns_)
    print(f"[rules] Encoded matrix: {encoded.shape}")

    print(f"[rules] FP-Growth (min_support={MIN_SUPPORT})")
    fp = fpgrowth(encoded, min_support=MIN_SUPPORT, use_colnames=True)
    fp = fp.sort_values("support", ascending=False).reset_index(drop=True)
    fp_out = fp.copy()
    fp_out["itemsets_str"] = fp_out["itemsets"].apply(fmt_set)
    fp_out["length"] = fp_out["itemsets"].apply(len)
    fp_out[["support", "length", "itemsets_str"]].to_csv(
        RESULTS / "frequent_itemsets_fpgrowth.csv", index=False)
    print(f"[rules] FP-Growth itemsets: {len(fp)}")
    print(fp_out.head(15)[["support", "length", "itemsets_str"]].to_string(index=False))

    print(f"\n[rules] Apriori (same min_support, max_len=3)")
    ap = apriori(encoded, min_support=MIN_SUPPORT, use_colnames=True, max_len=3)
    ap = ap.sort_values("support", ascending=False).reset_index(drop=True)
    ap_out = ap.copy()
    ap_out["itemsets_str"] = ap_out["itemsets"].apply(fmt_set)
    ap_out["length"] = ap_out["itemsets"].apply(len)
    ap_out[["support", "length", "itemsets_str"]].to_csv(
        RESULTS / "frequent_itemsets_apriori.csv", index=False)
    print(f"[rules] Apriori itemsets: {len(ap)}")

    print(f"\n[rules] Generating rules (min_confidence={MIN_CONFIDENCE})")
    rules = association_rules(fp, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    rules["antecedents_str"] = rules["antecedents"].apply(fmt_set)
    rules["consequents_str"] = rules["consequents"].apply(fmt_set)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    keep = ["antecedents_str", "consequents_str", "support", "confidence", "lift",
            "leverage", "conviction"]
    rules[keep].to_csv(RESULTS / "association_rules.csv", index=False)
    print(f"[rules] Total rules: {len(rules)}")
    print(rules.head(15)[["antecedents_str", "consequents_str", "support",
                          "confidence", "lift"]].to_string(index=False))
    print("[rules] Done.")


if __name__ == "__main__":
    main()
