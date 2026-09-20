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

idx = np.arange(len(original_df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_orig, random_state=42)
y_test = y_orig[test_idx]  # true labels for the held-out set - SAME across all 3 conditions

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

shap_orig = np.array(explainer.shap_values(X_test_orig))
shap_light = np.array(explainer.shap_values(X_test_light))
shap_heavy = np.array(explainer.shap_values(X_test_heavy))

def true_class_importance(shap_vals, y_true):
    # shap_vals shape: (n_samples, n_features, n_classes)
    # For each sample, pull ONLY the SHAP values for that sample's TRUE class
    n_samples, n_features, n_classes = shap_vals.shape
    true_class_shap = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        true_class_shap[i] = shap_vals[i, :, y_true[i]]
    return np.mean(np.abs(true_class_shap), axis=0)

imp_orig = true_class_importance(shap_orig, y_test)
imp_light = true_class_importance(shap_light, y_test)
imp_heavy = true_class_importance(shap_heavy, y_test)

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

pca_start = len(STYLO_COLS)
pca_orig_sum = np.sum(imp_orig[pca_start:])
pca_light_sum = np.sum(imp_light[pca_start:])
pca_heavy_sum = np.sum(imp_heavy[pca_start:])
pca_orig_mean = np.mean(imp_orig[pca_start:])
pca_light_mean = np.mean(imp_light[pca_start:])
pca_heavy_mean = np.mean(imp_heavy[pca_start:])

results.append({
    'feature': 'semantic_block_TOTAL (sum of all PCA components)',
    'type': 'semantic',
    'shap_original': pca_orig_sum,
    'shap_light': pca_light_sum,
    'shap_heavy': pca_heavy_sum,
    'survival_light_pct': (pca_light_sum / pca_orig_sum * 100) if pca_orig_sum > 0 else 0,
    'survival_heavy_pct': (pca_heavy_sum / pca_orig_sum * 100) if pca_orig_sum > 0 else 0,
})
results.append({
    'feature': 'semantic_PER-COMPONENT average (fair vs single stylometric feature)',
    'type': 'semantic',
    'shap_original': pca_orig_mean,
    'shap_light': pca_light_mean,
    'shap_heavy': pca_heavy_mean,
    'survival_light_pct': (pca_light_mean / pca_orig_mean * 100) if pca_orig_mean > 0 else 0,
    'survival_heavy_pct': (pca_heavy_mean / pca_orig_mean * 100) if pca_orig_mean > 0 else 0,
})

results_df = pd.DataFrame(results)
results_df.to_csv("shap_feature_survival_v2.csv", index=False)

print("\n=== TRUE-CLASS-RESTRICTED SHAP RESULTS ===\n")
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(results_df[['feature','type','shap_original','shap_light','shap_heavy',
                   'survival_light_pct','survival_heavy_pct']].to_string(index=False))

stylo_only = results_df[results_df['type']=='stylometric'].sort_values('shap_original', ascending=False)
print("\nStylometric features ranked by original importance:")
print(stylo_only[['feature','shap_original']].to_string(index=False))

stylo_by_survival = stylo_only.sort_values('survival_heavy_pct', ascending=False)
print("\nMost edit-resistant (highest heavy survival %):")
print(stylo_by_survival.head(3)[['feature','survival_heavy_pct']].to_string(index=False))
print("\nMost edit-sensitive (lowest heavy survival %):")
print(stylo_by_survival.tail(3)[['feature','survival_heavy_pct']].to_string(index=False))

print("\nSaved to shap_feature_survival_v2.csv")
