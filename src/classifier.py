import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "scam_model.pkl")
vectorizer_path = os.path.join(script_dir, "..", "models", "vectorizer.pkl")

#* Get model 
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#* Initialize classify function 
def classify(text):
    vec = vectorizer.transform([text])
    prediction = str(model.predict(vec)[0])
    probabilities = model.predict_proba(vec)[0]
    confidence = {str(cls): float(str(prob)) for cls, prob in zip(model.classes_, probabilities)}
    return {"label": prediction, "confidence": confidence}