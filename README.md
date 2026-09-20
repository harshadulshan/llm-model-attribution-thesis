# Which Model Leaves the Last Fingerprint?

**Stylometric Attribution Survival Under LLM-Based Rewriting Pressure**

An undergraduate thesis project (BSc Management Information Systems, NSBM Green University) investigating whether stylometric and semantic features can attribute AI-generated text to its source model — ChatGPT, Claude, Gemini, or Grok — and how well that attribution survives when the text is subsequently rewritten by another LLM.

## Research question

Given a piece of text known to be AI-generated, can a classifier identify *which* model produced it — and does that signal survive light (~40% edit distance) or heavy (~65% edit distance) LLM-based paraphrasing?

## Approach

- **Dataset**: 1,196 samples across 4 models (ChatGPT via RAID benchmark, Claude, Gemini, Grok) and 4 domains (abstracts, books, news, reddit).
- **Rewriting pipeline**: each sample rewritten at two intensities by an independent LLM, filtered through a dual-metric quality gate (word-level Levenshtein distance + cosine similarity on sentence embeddings) so only rewrites that actually hit the target edit intensity *and* preserved meaning are kept (609 samples pass both gates).
- **Features**: 8 stylometric features (sentence-length statistics, type-token ratio, hedging frequency, passive-voice rate, comma density, mean word length, Flesch-Kincaid grade) + 384-dim sentence embeddings (all-MiniLM-L6-v2) reduced to 75 components via PCA.
- **Classifier**: soft-voting ensemble of XGBoost, Random Forest, and an MLP, evaluated with 5-fold stratified cross-validation.
- **Evaluation**: trained only on unmodified (0% edit) text, then tested out-of-distribution on the light and heavy rewrites from the same held-out samples, to measure how much attribution signal survives rewriting.
- **Interpretability**: SHAP analysis on the ensemble's XGBoost component, restricted to each sample's true class, to identify which features carry the most signal and which survive rewriting best.

## Key results

| Condition | Macro F1 (mean ± std across 5 folds) |
|---|---|
| Baseline (0% edit) | 0.7061 ± 0.0234 |
| Light rewrite (~40% edit) | 0.5301 ± 0.0245 |
| Heavy rewrite (~65% edit) | 0.4017 ± 0.0231 |

(Chance baseline for 4 classes: 0.25)

Attribution accuracy degrades substantially but stays well above chance even under heavy rewriting — the strongest surviving signal comes from lexical diversity (type-token ratio) and sentence-length statistics, while comma density and hedging frequency are the most edit-resistant features individually.

## Repository structure

```
├── app.py                          # Gradio demo — paste text, get live model attribution
├── requirements.txt
├── generate_full_rewrites.py       # LLM-based rewriting pipeline with quality-gate retry logic
├── compute_cosine_gate.py          # Dual-metric (Levenshtein + cosine) quality gate
├── extract_features.py             # Stylometric feature extraction
├── extract_embeddings.py           # Semantic embedding extraction
├── train_classifier_v2.py          # 5-fold CV training (single fold saved)
├── train_all_folds.py              # Same training, saves all 5 fold pipelines (used by app.py)
├── ablation_study_v2.py            # Stylometric-only / semantic-only / combined ablation
├── shap_analysis_v3.py             # SHAP feature-survival analysis on the trained ensemble
├── confusion_matrices.py           # Confusion matrices for baseline/light/heavy
├── verify_alignment.py             # Data integrity checks
├── master_clean_originals.csv      # Source dataset (1,196 samples)
├── master_rewritten_samples_FINAL.csv
├── features_extracted.csv
├── semantic_embeddings.npy
├── fold1_pipeline.pkl … fold5_pipeline.pkl   # Trained model pipelines (one per CV fold)
└── *.png / *.csv                   # Confusion matrices, SHAP beeswarm, ablation results
```

## Running the demo locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

Opens a local Gradio interface where you can paste any text and get a live 4-way prediction, averaged across all 5 trained cross-validation folds.

## Limitations

Trained on a modest research dataset (609 gate-passing samples after quality filtering); predictions are illustrative of the research findings, not a certified detection tool. See the full thesis for methodology, related work, and discussion.
