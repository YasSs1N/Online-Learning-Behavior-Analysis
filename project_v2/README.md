# Online Learning Behavior Analysis (v2)
**Project #6 — Data Mining Course**

End-to-end implementation matching the rubric exactly: course-level pattern
mining + link analysis on real student multi-enrolments, plus BERT semantic
similarity over real course descriptions.

## Why two datasets

| Task | Dataset | Reason |
|---|---|---|
| Task 1 — FP-Growth + PageRank | **OULAD** (Open University Learning Analytics Dataset) | 32k+ student × module records, ~2,500 students with multi-enrolments → real "common course combinations" |
| Task 2 — BERT similarity | **Coursera Courses Dataset 2021** | 3,400+ courses with full descriptions (~1,160 chars avg) → real "BERT on course descriptions" |

No single public dataset has both, so we use the right one for each task and
disclose the methodology transparently in the report.

## Module mapping (OULAD)

OULAD anonymises modules as `AAA`–`GGG`. The Kuzilek et al. (2017) paper
discloses the 4 STEM + 3 Social Sciences split; subject-area names below are
inferred from each module's enrolment, gender, and assessment signature.

| Code | Module name | Domain |
|---|---|---|
| AAA | Social Sciences Foundation | Social Sciences |
| BBB | Humanities | Social Sciences |
| CCC | Computing & IT | STEM |
| DDD | Technology Studies | STEM |
| EEE | Mathematics | STEM |
| FFF | Engineering | STEM |
| GGG | Education Studies | Social Sciences |

The mapping is stored in `data/processed/oulad_module_mapping.xlsx` — the
single source of truth read by every script. **Edit that file to change
labels everywhere at once.**

## Rubric → deliverable map

| Rubric item | Marks | File |
|---|---|---|
| Data Collection & Understanding | 2 | `notebooks/...ipynb` §1, §6 |
| Data Preprocessing | 2 | `scripts/data_prep_oulad.py`, `scripts/data_prep_coursera.py` |
| FP-Growth & Apriori | 2 | `scripts/association_rules.py` → `results/association_rules.csv` |
| PageRank & HITS | 2 | `scripts/link_analysis.py` → `results/course_pagerank.csv`, `course_hits.csv` |
| Visualisation | 1 | `scripts/visualization.py` → `figures/01_…08_…png` |
| Report & Presentation | 1 | `report/...docx`, `slides/...pptx` |
| **BERT model (Task 2)** | **5** | `scripts/bert_similarity.py` → `results/bert_*.csv`, `bert_*.npz`, `bert_*.npy` |
| Final output (recommended paths) | bonus | `scripts/learning_paths.py` → `results/recommended_paths.csv` |

## Project layout

```
project_v2/
├── data/
│   ├── raw/
│   │   ├── oulad/        OULAD CSVs (studentInfo, studentRegistration, ...)
│   │   └── coursera/     Coursera.csv
│   └── processed/        oulad_module_mapping.xlsx, transactions.csv, ...
├── scripts/              data_prep_oulad.py, data_prep_coursera.py,
│                         association_rules.py, link_analysis.py,
│                         bert_similarity.py, learning_paths.py,
│                         visualization.py, build_notebook.py,
│                         build_report.py, build_slides.py
├── notebooks/            Online_Learning_Behavior_Analysis.ipynb
├── figures/              01_…08_*.png
├── results/              frequent_itemsets_*.csv, association_rules.csv,
│                         course_pagerank.csv, course_hits.csv,
│                         course_graph.graphml,
│                         bert_minilm_top_neighbours.csv,
│                         bert_base_top_neighbours.csv,
│                         bert_model_comparison.csv,
│                         recommended_paths.csv
├── report/               Online_Learning_Behavior_Analysis_Report.docx
├── slides/               Online_Learning_Behavior_Analysis.pptx
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn networkx mlxtend \
            openpyxl python-docx python-pptx jupyter nbformat \
            torch transformers sentence-transformers
```

`torch` + `transformers` + `sentence-transformers` are required for the real
Task 2 BERT outputs. If they aren't installed, `bert_similarity.py` falls back
to a deterministic feature-augmented projection so the pipeline still
produces valid figures and CSVs — install them locally for the real BERT
embeddings.

```bash
# From the project_v2 root
python scripts/data_prep_oulad.py            # OULAD preprocessing + co-enrolment graph
python scripts/data_prep_coursera.py         # Coursera cleaning + topic tagging
python scripts/association_rules.py          # FP-Growth + Apriori on real OULAD baskets
python scripts/link_analysis.py              # PageRank + HITS on the course graph
python scripts/bert_similarity.py --sample 500   # BERT (both encoders) on Coursera
python scripts/visualization.py              # All figures
python scripts/learning_paths.py             # Combined recommendations
python scripts/build_notebook.py             # Rebuild the .ipynb
python scripts/build_report.py               # Rebuild the .docx
python scripts/build_slides.py               # Rebuild the .pptx
```

Or open the notebook and run all cells — it calls the same scripts internally:

```bash
jupyter notebook notebooks/Online_Learning_Behavior_Analysis.ipynb
```

## Key findings

1. **Mathematics ↔ Computing & IT** is the strongest curricular bridge in
   OULAD: students who took one are 2.5× more likely than baseline to also
   take the other (lift = 2.52, support 3.7%).
2. **Computing & IT** is the platform's central module by PageRank, despite
   not being the largest by enrolment — centrality reflects connectivity,
   not size. HITS confirms it as both top hub and top authority.
3. **BERT clusters Coursera courses by content** and surfaces real course
   sequels automatically (Health Systems Development I/II/III, IoT Capstone
   v1/v2, Digital Signal Processing 3/4, Engineering Dynamics 2D/3D, Java
   teaching modules) — exactly the "common course sequences" the rubric asks
   for.
4. **Both BERT variants** (MiniLM + bert-base-uncased) agree at high Pearson
   correlation, confirming the embedding choice is robust.
5. The unified `recommended_paths.csv` is the project's final deliverable: a
   single table combining FP-Growth course rules, PageRank centrality, and
   BERT similarity into actionable learning-path suggestions.
