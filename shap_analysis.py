import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import shap

print("Loading features, embeddings, and fold 1's saved pipeline...")
df = pd.read_csv("features_extracted.csv")
embeddings = np.load("semantic_embeddings.npy")
df['emb_idx'] = range(len(df))

STYLO_COLS = ['mean_sentence_length','sentence_length_variance','ttr','hedging_frequency',
              'passive_voice_rate','comma_density','mean_word_length','flesch_kincaid']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['model_label'] = le.fit_transform(df['model'])

original_df = df[df['edit_condition'] == 'original'].reset_index(drop=True)
light_df = df[df['edit_condition'] == 'light'].reset_index(drop=True)
heavy_df = df[df['edit_condition'] == 'heavy'].reset_index(drop=True)

X_stylo_orig = original_df[STYLO_COLS].values
X_emb_orig = embeddings[original_df['emb_idx'].values]
y_orig = original_df['model_label'].values

X_stylo_light = light_df[STYLO_COLS].values
X_emb_light = embeddings[light_df['emb_idx'].values]

X_stylo_heavy = heavy_df[STYLO_COLS].values
X_emb_heavy = embeddings[heavy_df['emb_idx'].values]

with open("fold1_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

ensemble = pipeline['ensemble']
stylo_scaler = pipeline['stylo_scaler']
emb_scaler = pipeline['emb_scaler']
pca = pipeline['pca']
test_idx = pipeline['test_idx']

# Pull the ACTUAL fitted XGBoost sub-model out of the real trained ensemble
xgb_from_ensemble = ensemble.named_estimators_['xgb']
print(f"Extracted XGBoost from fold 1's real ensemble (test set size: {len(test_idx)})")

y_test = y_orig[test_idx]

def prep_condition(stylo_raw, emb_raw, idx_subset):
    stylo_s = stylo_scaler.transform(stylo_raw[idx_subset])
    emb_s = emb_scaler.transform(emb_raw[idx_subset])
    emb_pca = pca.transform(emb_s)
    return np.hstack([stylo_s, emb_pca])

X_test_orig = prep_condition(X_stylo_orig, X_emb_orig, test_idx)
X_test_light = prep_condition(X_stylo_light, X_emb_light, test_idx)
X_test_heavy = prep_condition(X_stylo_heavy, X_emb_heavy, test_idx)

print("Computing SHAP values on fold 1's actual ensemble XGBoost component...")
explainer = shap.TreeExplainer(xgb_from_ensemble)

shap_orig = np.array(explainer.shap_values(X_test_orig))
shap_light = np.array(explainer.shap_values(X_test_light))
shap_heavy = np.array(explainer.shap_values(X_test_heavy))

def true_class_importance(shap_vals, y_true):
    n_samples, n_features, n_classes = shap_vals.shape
    true_class_shap = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        true_class_shap[i] = shap_vals[i, :, y_true[i]]
    return np.mean(np.abs(true_class_shap), axis=0), true_class_shap

imp_orig, raw_orig = true_class_importance(shap_orig, y_test)
imp_light, raw_light = true_class_importance(shap_light, y_test)
imp_heavy, raw_heavy = true_class_importance(shap_heavy, y_test)

results = []
for i, name in enumerate(STYLO_COLS):
    results.append({
        'feature': name, 'type': 'stylometric',
        'shap_original': imp_orig[i], 'shap_light': imp_light[i], 'shap_heavy': imp_heavy[i],
        'survival_light_pct': (imp_light[i] / imp_orig[i] * 100) if imp_orig[i] > 0 else 0,
        'survival_heavy_pct': (imp_heavy[i] / imp_orig[i] * 100) if imp_orig[i] > 0 else 0,
    })

pca_start = len(STYLO_COLS)
pca_orig_sum, pca_light_sum, pca_heavy_sum = np.sum(imp_orig[pca_start:]), np.sum(imp_light[pca_start:]), np.sum(imp_heavy[pca_start:])
pca_orig_mean, pca_light_mean, pca_heavy_mean = np.mean(imp_orig[pca_start:]), np.mean(imp_light[pca_start:]), np.mean(imp_heavy[pca_start:])

results.append({'feature': 'semantic_block_TOTAL (sum of 75 PCA components)', 'type': 'semantic',
    'shap_original': pca_orig_sum, 'shap_light': pca_light_sum, 'shap_heavy': pca_heavy_sum,
    'survival_light_pct': (pca_light_sum/pca_orig_sum*100) if pca_orig_sum>0 else 0,
    'survival_heavy_pct': (pca_heavy_sum/pca_orig_sum*100) if pca_orig_sum>0 else 0})
results.append({'feature': 'semantic_PER-COMPONENT average', 'type': 'semantic',
    'shap_original': pca_orig_mean, 'shap_light': pca_light_mean, 'shap_heavy': pca_heavy_mean,
    'survival_light_pct': (pca_light_mean/pca_orig_mean*100) if pca_orig_mean>0 else 0,
    'survival_heavy_pct': (pca_heavy_mean/pca_orig_mean*100) if pca_orig_mean>0 else 0})

results_df = pd.DataFrame(results)
results_df.to_csv("shap_feature_survival_FINAL.csv", index=False)

print("\n=== SHAP RESULTS FROM REAL FOLD-1 ENSEMBLE XGBOOST (PCA=75, true-class-restricted) ===\n")
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(results_df[['feature','type','shap_original','shap_light','shap_heavy','survival_light_pct','survival_heavy_pct']].to_string(index=False))

stylo_only = results_df[results_df['type']=='stylometric'].sort_values('shap_original', ascending=False)
print("\nRanked by original importance:")
print(stylo_only[['feature','shap_original']].to_string(index=False))

by_surv = stylo_only.sort_values('survival_heavy_pct', ascending=False)
print("\nMost edit-resistant:")
print(by_surv.head(3)[['feature','survival_heavy_pct']].to_string(index=False))
print("\nMost edit-sensitive:")
print(by_surv.tail(3)[['feature','survival_heavy_pct']].to_string(index=False))
import matplotlib.pyplot as plt

X_test_orig_stylo_only = X_test_orig[:, :8]  # just the 8 named stylometric features, skip PCA columns
raw_orig_stylo_only = raw_orig[:, :8]

shap.summary_plot(raw_orig_stylo_only, X_test_orig_stylo_only,
                   feature_names=STYLO_COLS, show=False)
plt.title("Figure 4.3b: SHAP Beeswarm — Stylometric Features (Baseline, True-Class)")
plt.tight_layout()
plt.savefig("shap_beeswarm_baseline.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved shap_beeswarm_baseline.png")
print("\nSaved to shap_feature_survival_FINAL.csv")
