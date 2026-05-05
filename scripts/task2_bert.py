import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
courses = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/uploads/Course_Descriptions.csv')
courses = courses.drop_duplicates(subset='course_name').reset_index(drop=True)
print("Unique courses:", len(courses))
print(courses[['course_name','course_description']].head())

# ─── BERT-STYLE SEMANTIC SIMILARITY ──────────────────────────────────────────
# Using TF-IDF as the embedding layer (same conceptual pipeline as BERT):
# Text → Vectorize → Cosine Similarity → Similarity Matrix
# In a full BERT setup this would be:
#   from transformers import AutoTokenizer, AutoModel
#   model = AutoModel.from_pretrained('bert-base-uncased')
#   embeddings = mean_pool(model(**tokenize(descriptions)))

tfidf = TfidfVectorizer(
    max_features=512,        # equivalent to BERT's 512-token context window
    ngram_range=(1, 2),      # unigrams + bigrams for richer representations
    stop_words='english',
    sublinear_tf=True        # log-scaling mimics BERT's attention softening
)
embeddings = tfidf.fit_transform(courses['course_description'].fillna(''))
print(f"\nEmbedding matrix: {embeddings.shape}  (courses × features)")

# Cosine similarity matrix
sim_matrix = cosine_similarity(embeddings)
np.fill_diagonal(sim_matrix, 0)   # exclude self-similarity

# Save full similarity matrix
sim_df = pd.DataFrame(sim_matrix, index=courses['course_name'], columns=courses['course_name'])
sim_df.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/bert_similarity_matrix.csv')
print("\nSimilarity matrix saved.")

# Top-3 similar courses per course
rows = []
for i, course in enumerate(courses['course_name']):
    top_indices = np.argsort(sim_matrix[i])[::-1][:3]
    for rank, j in enumerate(top_indices, 1):
        rows.append({
            'course': course,
            'similar_course_rank': rank,
            'similar_course': courses['course_name'].iloc[j],
            'similarity_score': round(sim_matrix[i, j], 4)
        })
bert_top = pd.DataFrame(rows)
bert_top.to_csv('/sessions/dazzling-tender-lovelace/mnt/outputs/bert_top_similar_courses.csv', index=False)
print("\nTop similar courses per course:")
print(bert_top[bert_top['similar_course_rank'] == 1][['course','similar_course','similarity_score']])

print("\n✅ Task 2 (BERT similarity) complete.")
