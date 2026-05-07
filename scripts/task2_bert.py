import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
courses = pd.read_csv('/sessions/dazzling-tender-lovelace/mnt/uploads/Course_Descriptions.csv')
courses = courses.drop_duplicates(subset='course_name').reset_index(drop=True)
print("Unique courses:", len(courses))
print(courses[['course_name','course_description']].head())

# ─── BERT SEMANTIC SIMILARITY (bert-base-uncased) ────────────────────────────
# Pipeline: tokenize → BERT encode → mean-pool → cosine similarity

print("\n🔄 Loading bert-base-uncased (downloads ~440 MB on first run)...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model     = AutoModel.from_pretrained('bert-base-uncased')
model.eval()                                  # disable dropout for inference
print("✅ BERT loaded")

def get_bert_embedding(text):
    """Tokenize, encode, and mean-pool BERT to get one 768-dim vector per text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

embeddings = np.stack([
    get_bert_embedding(desc)
    for desc in courses['course_description'].fillna('')
])
print(f"\nEmbedding matrix: {embeddings.shape}  (courses × 768 BERT dims)")

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
