"""
Streamlit GUI — Course Learning Path Recommender
=================================================

Reads `recommended_learning_paths.csv` and lets the user slide through the list
of starting courses. When a course is picked, the app shows the full 3-step
recommended path, the metric breakdown that justifies it, and the rationale.

Run from the project root:
    streamlit run scripts/recommended_path_app.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── Locate the CSV regardless of where streamlit is launched from ──────
SCRIPT_DIR = Path(__file__).parent
CSV_PATH   = SCRIPT_DIR.parent / 'results' / 'recommended_learning_paths.csv'

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Learning Path Recommender",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Course Learning Path Recommender")
st.caption("Pick a starting course and get a 3-step learning path "
           "supported by PageRank, association rules, and BERT similarity.")

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_paths(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

try:
    paths = load_paths(CSV_PATH)
except FileNotFoundError:
    st.error(f"❌ Could not find {CSV_PATH.name} at:\n\n`{CSV_PATH}`\n\n"
             "Make sure you're running streamlit from the project root.")
    st.stop()

start_courses = paths['step_1_start_course'].tolist()

# ── Course picker (slider over course names) ───────────────────────────
st.subheader("1. Pick your starting course")
selected = st.select_slider(
    "Slide through the available start courses:",
    options=start_courses,
    value=start_courses[0],
)

# ── Pull the matching path row ─────────────────────────────────────────
row = paths[paths['step_1_start_course'] == selected].iloc[0]

# ── The 3-step path ────────────────────────────────────────────────────
st.subheader("2. Your recommended learning path")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🟢 Step 1 — Start**")
    st.success(row['step_1_start_course'])
with c2:
    st.markdown("**🟡 Step 2 — Next**")
    st.warning(row['step_2_next_course'])
with c3:
    st.markdown("**🔵 Step 3 — Advanced**")
    st.info(row['step_3_advanced_course'])

st.caption(f"📂 Domain: **{row['domain']}**")

# ── Metric breakdown ───────────────────────────────────────────────────
st.subheader("3. Why this path was chosen")

m1, m2, m3, m4 = st.columns(4)
m1.metric("PageRank score",       f"{row['pagerank_score']:.4f}")
m2.metric("Assoc. confidence",    f"{row['association_confidence']:.4f}")
m3.metric("BERT similarity",      f"{row['bert_similarity']:.4f}")
m4.metric("Composite score",      f"{row['composite_score']:.4f}")

# ── Score-component bar chart ──────────────────────────────────────────
st.markdown("**Composite score breakdown** "
            "(weights: PageRank × 0.5, Confidence × 0.3, BERT × 0.2)")
score_components = pd.DataFrame({
    'component': ['PageRank (×0.5)', 'Confidence (×0.3)', 'BERT (×0.2)'],
    'weighted_value': [
        0.5 * row['pagerank_score'],
        0.3 * row['association_confidence'],
        0.2 * row['bert_similarity'],
    ],
}).set_index('component')
st.bar_chart(score_components, height=220)

# ── Rationale text ─────────────────────────────────────────────────────
st.subheader("4. Recommendation rationale")
st.markdown(f"> {row['rationale']}")

# ── Skills taught (if column exists) ───────────────────────────────────
if 'skills_taught' in row and isinstance(row['skills_taught'], str):
    st.subheader("5. Skills you'll pick up along this path")
    skills = [s.strip() for s in row['skills_taught'].split('|') if s.strip()]
    cols = st.columns(3)
    for i, skill in enumerate(skills):
        cols[i % 3].markdown(f"- {skill}")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"Source: {CSV_PATH.name} · {len(paths)} recommended paths total")
