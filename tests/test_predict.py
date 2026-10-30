"""
A small, focused set of sanity checks for the trained model 

Requires models/scam_model.pkl + models/vectorizer.pkl to already
exist -- run src/train.py first if they're missing.
"""
import os

import joblib
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TESTS_DIR, "..", "models", "scam_model.pkl")
VECTORIZER_PATH = os.path.join(TESTS_DIR, "..", "models", "vectorizer.pkl")


@pytest.fixture(scope="module")
def model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        pytest.skip("models/ is empty -- run src/train.py first to generate the model")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def classify(model_and_vectorizer, text):
    model, vectorizer = model_and_vectorizer
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def test_obvious_fake_etransfer(model_and_vectorizer):
    text = "URGENT: Your Interac e-Transfer of $680 will be cancelled in 2 hours. Click now to deposit: interac-urgent-claim.com"
    assert classify(model_and_vectorizer, text) == "fake_etransfer"


def test_obvious_fake_cra(model_and_vectorizer):
    text = "This is the Canada Revenue Agency. You owe $3,200 in unpaid taxes. Failure to pay within 24 hours will result in legal action. Call 1-800-555-0199."
    assert classify(model_and_vectorizer, text) == "fake_cra"


def test_obvious_fake_canada_post(model_and_vectorizer):
    text = "CANADA POST: We attempted to deliver your parcel but failed. Reschedule delivery and pay $2.99 fee: cpost-tracking.info"
    assert classify(model_and_vectorizer, text) == "fake_canada_post"


def test_obvious_legit(model_and_vectorizer):
    text = "Hey, are we still meeting for coffee tomorrow at 10am?"
    assert classify(model_and_vectorizer, text) == "legit"


def test_tricky_legit_mentions_cra(model_and_vectorizer):
    # The genuinely hard case: a real, non-scam message that still mentions "CRA".
    text = "Your CRA Notice of Assessment for the 2025 tax year is now available in My Account. No action needed if you agree with it."
    assert classify(model_and_vectorizer, text) == "legit"
