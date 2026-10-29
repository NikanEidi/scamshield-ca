from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 
import joblib
import os


#* Build an absolute path to the CSV so it works no matter where this script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "scam_dataset.csv")

#* Load dataframe (Contains scam texts and labels)
df = pd.read_csv(csv_path)

#! Get firat top 5 Rows
# print(df.head())

#* Extract Featurs and Data
text = df['text']
label = df['label']

#* split train/test
X_train,X_test,y_train,y_test = train_test_split(
    text, label, test_size=0.33, random_state=42
)

#* Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train) # learn and convert string to number 
X_test_vec = vectorizer.transform(X_test) # only convert 

#* Train model 
model = MultinomialNB()
model.fit(X_train_vec, y_train)


#* Predict On DataSet
predictions = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

#* Predict on New Sentence 
new_review = ["Scotia Bank e-Transfer: $89.99 received. Click to deposit before it is returned to sender: scotiabank-deposit-now.net"]
new_vec = vectorizer.transform(new_review)
print("Prediction:", model.predict(new_vec))
print("Probability per class:", model.predict_proba(new_vec))

#* Save model 
model_path = os.path.join(script_dir, "..", "models", "scam_model.pkl")
vectorizer_path = os.path.join(script_dir, "..", "models", "vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)