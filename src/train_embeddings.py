import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#* Build an absolute path to the CSV so it works no matter where this script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "scam_dataset.csv")

#* Load dataframe (Contains scam texts and labels)
df = pd.read_csv(csv_path)

#* Extract features and labels
text = df["text"]
label = df["label"]

#* Split train/test (same split logic as train.py, so results are comparable)
X_train, X_test, y_train, y_test = train_test_split(
    text, label, test_size=0.33, random_state=42
)

#* Embed: turn each message into a 384-dim semantic vector instead of a TF-IDF word-count vector
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_train_vec = embedder.encode(X_train.tolist())
X_test_vec = embedder.encode(X_test.tolist())

#* Train model (Logistic Regression instead of Naive Bayes -- embeddings can be
#* negative numbers, which MultinomialNB can't handle, but LogisticRegression can)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

#* Predict on test set
predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

#* Predict on the same new message used in train.py, for a direct comparison
new_message = ["Scotia Bank e-Transfer: $89.99 received. Click to deposit before it is returned to sender: scotiabank-deposit-now.net"]
new_vec = embedder.encode(new_message)
print("Prediction:", model.predict(new_vec))
print("Probability per class:", model.predict_proba(new_vec))
