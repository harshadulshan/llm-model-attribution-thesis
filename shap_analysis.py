import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import shap

print("Loading features and embeddings...")
df = pd.read_csv("features_extracted.csv")
embeddings = np.load("semantic_embeddings.npy")
df['emb_idx'] = range(len(df))

STYLO_COLS = ['mean_sentence_length','sentence_length_variance','ttr','hedging_frequency',
              'passive_voice_rate','comma_density','mean_word_length','flesch_kincaid']

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

# 80/20 split - train on 80%, compute SHAP on the same held-out 20% across all 3 conditions
idx = np.arange(len(original_df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_orig, random_state=42)

print(f"Training on {len(train_idx)} samples, SHAP evaluation on {len(test_idx)} held-out samples")

stylo_scaler = StandardScaler().fit(X_stylo_orig[train_idx])
emb_scaler = StandardScaler().fit(X_emb_orig[train_idx])

stylo_train_s = stylo_scaler.transform(X_stylo_orig[train_idx])
emb_train_s = emb_scaler.transform(X_emb_orig[train_idx])

pca = PCA(n_components=0.95, random_state=42).fit(emb_train_s)
emb_train_pca = pca.transform(emb_train_s)
n_pca = pca.n_components_
print(f"PCA components retained: {n_pca}")

X_train = np.hstack([stylo_train_s, emb_train_pca])
y_train = y_orig[train_idx]

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                     random_state=42, eval_metric='mlogloss')
xgb.fit(X_train, y_train)

feature_names = STYLO_COLS + [f'pca_{i}' for i in range(n_pca)]

def prep_condition(stylo_raw, emb_raw, idx_subset):
    stylo_s = stylo_scaler.transform(stylo_raw[idx_subset])
    emb_s = emb_scaler.transform(emb_raw[idx_subset])
    emb_pca = pca.transform(emb_s)
    return np.hstack([stylo_s, emb_pca])

X_test_orig = prep_condition(X_stylo_orig, X_emb_orig, test_idx)
X_test_light = prep_condition(X_stylo_light, X_emb_light, test_idx)
X_test_heavy = prep_condition(X_stylo_heavy, X_emb_heavy, test_idx)

print("Computing SHAP values (TreeExplainer) for original, light, heavy...")
explainer = shap.TreeExplainer(xgb)

shap_orig = explainer.shap_values(X_test_orig)
shap_light = explainer.shap_values(X_test_light)
shap_heavy = explainer.shap_values(X_test_heavy)

# shap_values shape for multiclass: (n_samples, n_features, n_classes) or list per class depending on version
def mean_abs_importance(shap_vals):
    arr = np.array(shap_vals)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> average abs over samples and classes
        return np.mean(np.abs(arr), axis=(0, 2))
    else:
        return np.mean(np.abs(arr), axis=0)

imp_orig = mean_abs_importance(shap_orig)
imp_light = mean_abs_importance(shap_light)
imp_heavy = mean_abs_importance(shap_heavy)

results = []
for i, name in enumerate(STYLO_COLS):
    results.append({
        'feature': name,
        'type': 'stylometric',
        'shap_original': imp_orig[i],
        'shap_light': imp_light[i],
        'shap_heavy': imp_heavy[i],
        'survival_light_pct': (imp_light[i] / imp_orig[i] * 100) if imp_orig[i] > 0 else 0,
        'survival_heavy_pct': (imp_heavy[i] / imp_orig[i] * 100) if imp_orig[i] > 0 else 0,
    })

pca_orig_sum = np.sum(imp_orig[len(STYLO_COLS):])
pca_light_sum = np.sum(imp_light[len(STYLO_COLS):])
pca_heavy_sum = np.sum(imp_heavy[len(STYLO_COLS):])
results.append({
    'feature': 'semantic_aggregate (all PCA components)',
    'type': 'semantic',
    'shap_original': pca_orig_sum,
    'shap_light': pca_light_sum,
    'shap_heavy': pca_heavy_sum,
    'survival_light_pct': (pca_light_sum / pca_orig_sum * 100) if pca_orig_sum > 0 else 0,
    'survival_heavy_pct': (pca_heavy_sum / pca_orig_sum * 100) if pca_orig_sum > 0 else 0,
})

results_df = pd.DataFrame(results).sort_values('shap_original', ascending=False)
results_df.to_csv("shap_feature_survival.csv", index=False)

print("\n=== SHAP FEATURE SURVIVAL RESULTS (ranked by original importance) ===\n")
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(results_df[['feature','type','shap_original','shap_light','shap_heavy',
                   'survival_light_pct','survival_heavy_pct']].to_string(index=False))

print("\n\nTop 3 most edit-resistant stylometric features (highest heavy survival %):")
stylo_only = results_df[results_df['type']=='stylometric'].sort_values('survival_heavy_pct', ascending=False)
print(stylo_only.head(3)[['feature','survival_heavy_pct']].to_string(index=False))

print("\nBottom 3 most edit-sensitive stylometric features (lowest heavy survival %):")
print(stylo_only.tail(3)[['feature','survival_heavy_pct']].to_string(index=False))

print("\nSaved full results to shap_feature_survival.csv")
