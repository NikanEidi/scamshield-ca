from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd 
import os

#* Inatialize the absloute path
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "scam_dataset.csv")


#* add Model 
model = SentenceTransformer("all-MiniLM-L6-v2")

#* Load dataframe (Contains scam texts and labels)
df = pd.read_csv(csv_path)

t_data = df["text"]
label = df["label"]

#* Convert text to number
df_embeddings = model.encode(t_data.tolist())

#* Declare similar scam finder function 
def find_similar_scams(text, k=3):
    #* encode input text(new message)
    m_embeddings = model.encode([text])
    #* cosine similarity (get similarities between scams from dataSet and New input)
    similarities = cosine_similarity(m_embeddings, df_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for i in top_indices:
        results.append({
            "text": t_data[i],
            "label": label[i],
            "similarity": float(similarities[i]),
        })
    return results


