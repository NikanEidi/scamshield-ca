"""
A small, focused set of sanity checks for the trained model -- one
clear-cut case per class (now 5: fake_etransfer, fake_cra,
fake_canada_post, fake_bank_alert, legit), plus the two genuinely
hard cases: a legit message that mentions "CRA" for real, and a
legit bank security notification that looks like the fake_bank_alert
pattern but isn't asking you to click anything.

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


def test_obvious_fake_bank_alert(model_and_vectorizer):
    text = "TD Alert: Your account has been suspended due to unusual activity. Verify immediately: td-secure-verify.com"
    assert classify(model_and_vectorizer, text) == "fake_bank_alert"


def test_tricky_legit_mentions_cra(model_and_vectorizer):
    # The genuinely hard case: a real, non-scam message that still mentions "CRA".
    text = "Your CRA Notice of Assessment for the 2025 tax year is now available in My Account. No action needed if you agree with it."
    assert classify(model_and_vectorizer, text) == "legit"


def test_tricky_legit_real_security_notification(model_and_vectorizer):
    # A real bank security notification -- looks similar in topic to fake_bank_alert
    # but isn't asking the recipient to click a link or verify anything.
    text = "RBC: A new sign-in to your account was detected from your iPhone just now. If this was you, no action is needed."
    assert classify(model_and_vectorizer, text) == "legit"


def test_marketplace_overpayment_is_fake_etransfer(model_and_vectorizer):
    # The "I accidentally overpaid you, send back the difference" social-engineering
    # variant -- still an e-Transfer scam, just a different angle than "click to deposit".
    text = "Hi, so sorry, I think I sent you an extra $100 by e-Transfer for the couch by mistake. Could you send the difference back to this account?"
    assert classify(model_and_vectorizer, text) == "fake_etransfer"


def test_obvious_fake_bank_alert_different_bank(model_and_vectorizer):
    # Same class as test_obvious_fake_bank_alert, different bank + phrasing,
    # to make sure the model learned the pattern and not just the word "TD".
    text = "RBC: Suspicious login detected on your account. Confirm your identity within 24 hours to avoid permanent lock: rbc-account-restore.net"
    assert classify(model_and_vectorizer, text) == "fake_bank_alert"


def test_tricky_legit_otp_code(model_and_vectorizer):
    # A real one-time-passcode text -- mentions "security" and a bank name like
    # fake_bank_alert does, but is informational, not asking for any action.
    text = "TD Security: Your one-time passcode is 384719. Don't share this code with anyone."
    assert classify(model_and_vectorizer, text) == "legit"


def test_canada_post_incomplete_address_variant(model_and_vectorizer):
    # A different fake_canada_post angle than "pay a fee" -- the "incomplete
    # address, confirm here" pattern documented by Canada Post's own phishing guidance.
    text = "Canada Post: We could not deliver your parcel due to an incomplete address. Confirm your address here to reschedule: canadapost-address-confirm.com"
    assert classify(model_and_vectorizer, text) == "fake_canada_post"
