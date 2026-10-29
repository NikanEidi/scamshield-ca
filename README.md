# scamshield-ca

Text classifier that detects common Canadian scam messages (fake Interac e-Transfer, fake CRA tax notices, fake Canada Post delivery fees) and flags legitimate messages — with a confidence score per class.

## Why this project

Every Canadian recruiter has personally received one of these scam texts — makes the problem instantly understandable in an interview, unlike a generic ML demo dataset.

## Project structure

```
scamshield-ca/
├── data/
│   ├── scam_dataset.csv     # labeled dataset: text, label (fake_etransfer / fake_cra / fake_canada_post / legit)
│   └── generate_dataset.py  # builds scam_dataset.csv from templates (reproducible, fixed random seed)
├── src/
│   ├── train.py             # load → split → vectorize (TF-IDF) → train (Naive Bayes) → evaluate → save model
│   └── predict.py           # load saved model, classify a new message, print class + probabilities
├── models/                  # trained model artifacts (gitignored — regenerate via train.py)
├── notes/                   # written explanations of the concepts used (TF-IDF, Naive Bayes, pipeline)
├── assets/                  # hand-drawn pipeline diagram + model result screenshots
├── requirements.txt
└── .gitignore
```

My own hand-drawn pipeline diagram, mapped onto the actual folder structure:

![Project structure diagram](assets/ProjectStructure.jpg)

## Pipeline

```mermaid
flowchart TD
    Start([Start]) --> Collect["Collect data\n(text + label)"]
    Collect --> Preprocess["Preprocess\nremove chars, lowercase,\nextract text + labels"]
    Preprocess --> Split["Train / Test split\nTrain 70% · Test 30%"]
    Split --> Vectorize["Vectorize\nTF-IDF"]
    Vectorize --> Train["Train model\nMultinomial Naive Bayes"]
    Train --> Predict["Predict on test set"]
    Predict --> Evaluate["Evaluate\naccuracy, precision,\nrecall, f1"]
    Evaluate --> End([End])
```

See `notes/` for the full written explanation of each step, TF-IDF, and Naive Bayes.

## Results

Trained three times as the dataset grew — accuracy and confidence both climbed with more data:

| Dataset size | Accuracy | New-message confidence | Screenshot |
|---|---|---|---|
| 94 rows (hand-written) | 84% | 51% | `assets/first-model-results.png` |
| 241 rows (templated, ~60/class) | 96% | 80.5% | `assets/second-model-results.png` |
| 1000 rows (templated + real-scam patterns, 250/class) | **99%** | **92.6%** | `assets/third-model-results.png` |

Final classification report (1000-row dataset):

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

## Status

- [x] Project structure + dataset (1000 labeled examples, 4 classes, template-generated from real scam patterns)
- [x] `src/train.py` — training pipeline, 99% accuracy
- [x] Model persistence (joblib)
- [ ] `src/predict.py` — clean inference script on new messages
- [ ] (Later) LLM-generated plain-English explanation of why a message was flagged
