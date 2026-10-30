import joblib
import os

#* Build an absolute path to models/ so it works no matter where this script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "scam_model.pkl")
vectorizer_path = os.path.join(script_dir, "..", "models", "vectorizer.pkl")

#* Load the saved model + vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#* A new message to classify 
new_message = ["CRA Notice: You are eligible for a tax refund of $412.50. Claim now: cra-refund-claim.net"]

#* Vectorize with the SAME vectorizer used in training (transform only, not fit_transform)
new_vec = vectorizer.transform(new_message)

#* Predict the class + per-class probabilities
prediction = model.predict(new_vec)
probabilities = model.predict_proba(new_vec)

print("Message:", new_message[0])
print("Prediction:", prediction[0])
print("Confidence per class:")
for cls, prob in zip(model.classes_, probabilities[0]):
    print(f"  {cls}: {prob:.1%}")
