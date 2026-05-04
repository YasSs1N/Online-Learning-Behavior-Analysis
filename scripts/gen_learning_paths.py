import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

rules = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/association_rules.csv')
pagerank = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/pagerank_scores.csv')
bert = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/bert_top_similar_courses.csv')

pr = pagerank.set_index('course_name')['pagerank_score']
pr_norm = (pr - pr.min()) / (pr.max() - pr.min())
top_courses = pagerank['course_name'].head(10).tolist()

path_rows = []
for start_course in top_courses:
    step2_rules = rules[rules['antecedents'].str.contains(start_course, regex=False)]
    if len(step2_rules) > 0:
        step2_row = step2_rules.sort_values('confidence', ascending=False).iloc[0]
        step2 = step2_row['consequents'].replace("frozenset({'", "").replace("'})", "").strip()
        assoc_conf = round(float(step2_row['confidence']), 4)
    else:
        step2 = [c for c in top_courses if c != start_course][0]
        assoc_conf = 0.0

    bert_step3 = bert[
        (bert['course'] == step2) &
        (bert['similar_course'] != start_course) &
        (bert['similar_course'] != step2)
    ].sort_values('similarity_score', ascending=False)
    if len(bert_step3) > 0:
        step3 = bert_step3.iloc[0]['similar_course']
        bert_score = round(float(bert_step3.iloc[0]['similarity_score']), 4)
    else:
        step3 = "Self-directed exploration"
        bert_score = 0.0

    pr_score = float(pr_norm.get(start_course, 0))
    composite = round(0.5 * pr_score + 0.3 * assoc_conf + 0.2 * bert_score, 4)

    if any(w in start_course.lower() for w in ['computer','cs50','programming','circuit','electric','solid','structure','mechanic','biology']):
        domain = 'STEM / Computer Science'
    else:
        domain = 'Health & Social Sciences'

    pr_rank = int(pagerank[pagerank['course_name'] == start_course]['rank'].values[0])
    path_rows.append({
        'path_id': len(path_rows) + 1,
        'domain': domain,
        'step_1_start_course': start_course,
        'step_2_next_course': step2,
        'step_3_advanced_course': step3,
        'pagerank_score': round(float(pr.get(start_course, 0)), 6),
        'association_confidence': assoc_conf,
        'bert_similarity': bert_score,
        'composite_score': composite,
        'rationale': (f"Start with '{start_course}' (PageRank #{pr_rank}). "
                      f"Students commonly move to '{step2}' (confidence={assoc_conf}). "
                      f"BERT identifies '{step3}' as semantically aligned (similarity={bert_score}).")
    })

paths_df = pd.DataFrame(path_rows)
paths_df.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/recommended_learning_paths.csv', index=False)
print(paths_df[['path_id','domain','step_1_start_course','step_2_next_course','step_3_advanced_course','composite_score']].to_string())
print(f"\n✅ {len(paths_df)} learning paths saved.")
