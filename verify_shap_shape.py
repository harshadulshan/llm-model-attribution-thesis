import pickle
import numpy as np
import pandas as pd
import shap

with open("fold1_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)
ensemble = pipeline['ensemble']
xgb_from_ensemble = ensemble.named_estimators_['xgb']

df = pd.read_csv("features_extracted.csv")
embeddings = np.load("semantic_embeddings.npy")
df['emb_idx'] = range(len(df))
original_df = df[df['edit_condition'] == 'original'].reset_index(drop=True)

STYLO_COLS = ['mean_sentence_length','sentence_length_variance','ttr','hedging_frequency',
              'passive_voice_rate','comma_density','mean_word_length','flesch_kincaid']

test_idx = pipeline['test_idx']
stylo_scaler = pipeline['stylo_scaler']
emb_scaler = pipeline['emb_scaler']
pca = pipeline['pca']

X_stylo = stylo_scaler.transform(original_df[STYLO_COLS].values[test_idx])
X_emb = pca.transform(emb_scaler.transform(embeddings[original_df['emb_idx'].values[test_idx]]))
X_test = np.hstack([X_stylo, X_emb])

explainer = shap.TreeExplainer(xgb_from_ensemble)
shap_vals = explainer.shap_values(X_test)

print("Type returned:", type(shap_vals))
if isinstance(shap_vals, list):
    print(f"Returned as LIST of {len(shap_vals)} arrays, each shape: {shap_vals[0].shape}")
else:
    print(f"Returned as single array, shape: {shap_vals.shape}")
print(f"\nExpected: n_samples=153, n_features=83, n_classes=4")
