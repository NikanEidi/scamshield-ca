# Scam Shield — Log and Structure

## What this is

A text classifier that spots common Canadian scam messages — fake Interac e-Transfer notifications, fake CRA tax notices, fake Canada Post delivery-fee texts — and tells legit messages apart from them, with a confidence score per class. Picked this problem on purpose: every Canadian recruiter has personally gotten one of these scam texts, so it lands instantly in an interview instead of needing a generic ML-demo explanation.

## My own pipeline diagram

Drew this out by hand first to make sure I actually understood the flow before writing any code:

![My hand-drawn pipeline diagram](../assets/ProjectStructure.jpg)

Collect data → Preprocess → Train/Test split (70/30) → Vectorize (TF-IDF) → Train model (Naive Bayes) → Predict → Evaluate. Every box in that drawing maps to a real line (or a few lines) in `src/train.py`.

## How the code maps to the diagram

```
data/scam_dataset.csv     — the labeled data (text, label)
data/generate_dataset.py  — builds the dataset from templates, fixed random seed so it's reproducible
src/train.py               — the full pipeline: load → split → TF-IDF → Naive Bayes → evaluate → save
src/predict.py              — (next) loads the saved model, classifies a new message
models/                     — trained model + vectorizer, saved with joblib
```

## TF-IDF, in one line

Turns a sentence into a row of numbers — words that repeat a lot *in one message* but are rare *across the whole dataset* (like "etransfer-urgent-claim" or a fake domain) get a high score, because they're the informative ones. Words everyone uses (the, and, to) get pushed down to near-zero.

## Naive Bayes, in one line

A probability model that, for each class, learned during training how often each word shows up in messages of that class. For a new message, it multiplies together the probability of seeing each of its words under each class, and picks whichever class comes out most likely. "Naive" because it pretends every word is independent of the others — not really true, but it still works well for text.

## The dataset's three versions

I didn't get to 99% accuracy on the first try — the dataset itself had to grow up:

**v1 — 94 rows, hand-written.** First pass, written mostly by hand with some realistic variety. Accuracy came out to 84%, and a new test message only got 51% confidence. The real problem showed up in the classification report: `fake_canada_post` only had 3 examples in the test set, so its precision/recall numbers were basically noise — one wrong guess there swings the percentage hugely.

![First model results — 94 rows, 84% accuracy](../assets/first-model-results.png)

**v2 — 241 rows, templated.** Instead of typing more sentences by hand, I built `generate_dataset.py`: a handful of message templates (`"{bank} e-Transfer: ${amt} received..."`) filled in with different banks, amounts, and fake domains, with a fixed `random_state=42` seed so it regenerates identically every time. Accuracy jumped to 96%, confidence on the same test message to 80.5%.

![Second model results — 241 rows, 96% accuracy](../assets/second-model-results.png)

**v3 — 1000 rows, researched.** Searched real Canadian fraud-alert sources (CIBC's fraud-examples page, the CRA's own scam alerts, Canada Post's phishing guidance) and found the actual patterns scammers use — "reply INFO/YES/HELP" calls-to-action, a fake cashback-via-e-Transfer scam, "incomplete address, confirm here" Canada Post texts. Rebuilt the templates around those real patterns and scaled up to 250 rows per class (1000 total, perfectly balanced). Accuracy hit 99%, confidence on the same test message hit 92.6%.

![Third model results — 1000 rows, 99% accuracy](../assets/third-model-results.png)

## The one metric that tells the real story: `legit` recall

Across all three versions, `legit` was the class that struggled — not because legit messages are hard in general, but because I deliberately seeded some legit examples that *mention* "CRA" or "Interac" in a totally real way (e.g. "Your CRA Notice of Assessment is now available", "I just sent you $30 by Interac e-Transfer for dinner"). Those are the genuinely hard cases — a model that just pattern-matches on the word "CRA" would get them wrong every time.

- v1: `legit` recall = 0.86 (some of these tricky real messages got misclassified as scams)
- v3: `legit` recall = 1.00 (more real examples of this exact pattern taught the model the actual difference)

That's the clearest proof the dataset growth actually mattered, not just the raw accuracy number going up.

## predict.py and a focused test suite

Wrote `src/predict.py`: load the saved model + vectorizer with `joblib.load()`, `transform()` (not `fit_transform()`) a new message with that same vectorizer, then `model.predict()` + `model.predict_proba()` for the class and the confidence per class. Hit the same `os.path` relative-path bug a third time on the model/vectorizer paths and fixed it myself the same way as before.

The actual point of this step: understanding *why* you save a model at all. Without `joblib.dump`/`joblib.load`, classifying one new message means re-running the entire ~50-line training pipeline (load data, split, vectorize, train) every single time — train once, save to disk, then every future run just loads it. That clicked fully, not just the syntax.

Added `tests/test_predict.py` — 5 focused pytest cases (one obvious example per class, plus the genuinely hard case: a legit message that mentions "CRA" for real, not as a scam). All 5 pass. Deliberately kept it small — not a giant edge-case suite, just enough to catch a real regression if the model or dataset changes later.

## Turning it into an MCP server

Wanted the classifier to be usable directly inside Claude, not just from a script — so it needed to become an MCP tool. Two pieces:

**`src/classifier.py`** — pulled the inference logic out of `predict.py` into a reusable function so a long-running server process doesn't reload the model on every call. Model + vectorizer are loaded once at module level (outside any function); `classify(text)` just reuses them. Learned the hard way that `model.predict()` and `model.classes_` return `numpy` types (`np.str_`, `np.float64`), not plain Python `str`/`float` — harmless in a script, but breaks JSON serialization, which an MCP tool needs. Wrapped the label in `str(...)` and each confidence value in `float(...)`.

**`server/mcp_server.py`** — a thin wrapper using the `mcp` SDK (v2: `MCPServer`, not the older `FastMCP` name most tutorials still show). One tool, `classify_message`, registered with the `@mcp.tool()` decorator — first real use of a decorator in this project. The docstring on that function is what Claude actually reads to know what the tool does, so it has to describe it clearly, not just label it.

Smoke-tested by importing the module directly and calling `classify_message()` without starting the stdio server loop — confirmed it returns a clean, JSON-safe dict (`{'label': 'fake_cra', 'confidence': {...}}`).

## What's still open

- Wire up `.mcp.json` so Claude Code can actually discover and call `classify_message` from inside a real chat, then verify it end-to-end
- Stretch goal: an LLM-generated plain-English explanation of *why* a message got flagged, reusing the Groq-call pattern from the Nik-Blog Assistant RAG chatbot

Related: [Pipeline overview](01-pipeline-overview.md) · [TF-IDF and Naive Bayes](02-tfidf-and-naive-bayes.md) · [Dataset and results](03-dataset-and-results.md)
