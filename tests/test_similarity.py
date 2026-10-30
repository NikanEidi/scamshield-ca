"""
Sanity checks for find_similar_scams(): does it return the right number
of results, in the right shape, and are the closest matches actually
the same class as the message we searched for.
"""
from src.similarity import find_similar_scams


def test_returns_k_results():
    results = find_similar_scams("RBC Alert: Suspicious login detected on your account.", k=3)
    assert len(results) == 3


def test_result_shape():
    results = find_similar_scams("RBC Alert: Suspicious login detected on your account.", k=1)
    result = results[0]
    assert set(result.keys()) == {"text", "label", "similarity"}
    assert isinstance(result["text"], str)
    assert isinstance(result["label"], str)
    assert isinstance(result["similarity"], float)


def test_results_sorted_by_similarity_descending():
    results = find_similar_scams("RBC Alert: Suspicious login detected on your account.", k=5)
    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_closest_matches_are_the_same_scam_type():
    # A clear fake_bank_alert message -- the top matches should mostly be the same class.
    text = "TD Alert: Your account has been suspended due to unusual activity. Verify immediately: td-secure-verify.com"
    results = find_similar_scams(text, k=3)
    labels = [r["label"] for r in results]
    assert labels.count("fake_bank_alert") >= 2
