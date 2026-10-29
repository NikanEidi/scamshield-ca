# Dataset and results

## How the dataset grew

| Version | Size | How it was built |
|---|---|---|
| v1 | 94 rows | Hand-written by Claude: ~15–30 examples per class, realistic but limited variety |
| v2 | 241 rows | `data/generate_dataset.py` — template + placeholder generation (bank names, amounts, fake domains), fixed `random_state=42` seed |
| v3 | 1000 rows | Same generator, expanded templates based on patterns from real Canadian fraud-alert sources (CIBC, CRA, Canada Post), `PER_CLASS = 250` |

## Why `support` matters (and why v1 was misleading)

`support` = how many real examples of a class were actually in the test set. On v1, `fake_canada_post` only had **3** test examples — one wrong guess there swings precision/recall by huge amounts. By v3, every class has **80+** test examples, so the accuracy/precision/recall numbers are actually trustworthy.

## Accuracy and confidence over time

| Dataset size | Accuracy | New-message confidence (same test sentence each time) |
|---|---|---|
| 94 rows | 84% | 51% |
| 241 rows | 96% | 80.5% |
| 1000 rows | **99%** | **92.6%** |

Screenshots of each run are in `assets/first-model-results.png`, `second-model-results.png`, `third-model-results.png`.

## Final v3 classification report

```
                  precision    recall  f1-score   support

fake_canada_post       1.00      1.00      1.00        80
        fake_cra       1.00      0.96      0.98        85
  fake_etransfer       0.96      1.00      0.98        82
           legit       1.00      1.00      1.00        83

        accuracy                           0.99       330
       macro avg       0.99      0.99      0.99       330
    weighted avg       0.99      0.99      0.99       330
```

`legit` went from the weakest class (recall 0.86 on v1, because some legit messages deliberately mention "CRA" or "Interac" in a real, non-scam way) to a perfect 1.00 by v3 — more training examples of that tricky "mentions the keyword but is legit" pattern taught the model the difference.

Related: [Pipeline overview](01-pipeline-overview.md) · [TF-IDF and Naive Bayes](02-tfidf-and-naive-bayes.md)
