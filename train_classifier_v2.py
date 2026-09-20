import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import pickle

print("Loading features and embeddings...")
df = pd.read_csv("features_extracted.csv")
embeddings = np.load("semantic_embeddings.npy")
df['emb_idx'] = range(len(df))

STYLO_COLS = ['mean_sentence_length','sentence_length_variance','ttr','hedging_frequency',
              'passive_voice_rate','comma_density','mean_word_length','flesch_kincaid']

le = LabelEncoder()
df['model_label'] = le.fit_transform(df['model'])
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

original_df = df[df['edit_condition'] == 'original'].reset_index(drop=True)
light_df = df[df['edit_condition'] == 'light'].reset_index(drop=True)
heavy_df = df[df['edit_condition'] == 'heavy'].reset_index(drop=True)

X_stylo_orig = original_df[STYLO_COLS].values
X_emb_orig = embeddings[original_df['emb_idx'].values]
y_orig = original_df['model_label'].values

X_stylo_light = light_df[STYLO_COLS].values
X_emb_light = embeddings[light_df['emb_idx'].values]
y_light = light_df['model_label'].values

X_stylo_heavy = heavy_df[STYLO_COLS].values
X_emb_heavy = embeddings[heavy_df['emb_idx'].values]
y_heavy = heavy_df['model_label'].values

N_PCA = 75  # FIXED at 75 to match Chapter 2/3's stated ~83-dim vector and 11.6:1 ratio argument

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_f1_scores = []
ood_light_f1_scores = []
ood_heavy_f1_scores = []
fold_pipelines = []  # save each fold's fitted pieces for reuse by the SHAP script

print(f"\n=== Running 5-fold stratified CV (trained on 0% original only, PCA fixed at {N_PCA} components) ===")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_stylo_orig, y_orig), 1):
    stylo_train, stylo_test = X_stylo_orig[train_idx], X_stylo_orig[test_idx]
    emb_train, emb_test = X_emb_orig[train_idx], X_emb_orig[test_idx]
    y_train, y_test = y_orig[train_idx], y_orig[test_idx]

    stylo_scaler = StandardScaler().fit(stylo_train)
    emb_scaler = StandardScaler().fit(emb_train)

    stylo_train_scaled = stylo_scaler.transform(stylo_train)
    stylo_test_scaled = stylo_scaler.transform(stylo_test)
    emb_train_scaled = emb_scaler.transform(emb_train)
    emb_test_scaled = emb_scaler.transform(emb_test)

    pca = PCA(n_components=N_PCA, random_state=42).fit(emb_train_scaled)
    emb_train_pca = pca.transform(emb_train_scaled)
    emb_test_pca = pca.transform(emb_test_scaled)

    X_train_final = np.hstack([stylo_train_scaled, emb_train_pca])
    X_test_final = np.hstack([stylo_test_scaled, emb_test_pca])

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         random_state=42, eval_metric='mlogloss')
    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), alpha=0.01, max_iter=300, random_state=42)

    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)], voting='soft')
    ensemble.fit(X_train_final, y_train)

    preds = ensemble.predict(X_test_final)
    fold_f1 = f1_score(y_test, preds, average='macro')
    fold_f1_scores.append(fold_f1)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Fold {fold}: macro F1 = {fold_f1:.4f} (PCA={N_PCA} comps, {explained_var*100:.1f}% variance retained)")

    light_stylo_scaled = stylo_scaler.transform(X_stylo_light[test_idx])
    light_emb_scaled = emb_scaler.transform(X_emb_light[test_idx])
    light_emb_pca = pca.transform(light_emb_scaled)
    X_light_final = np.hstack([light_stylo_scaled, light_emb_pca])
    light_preds = ensemble.predict(X_light_final)
    ood_light_f1 = f1_score(y_light[test_idx], light_preds, average='macro')
    ood_light_f1_scores.append(ood_light_f1)

    heavy_stylo_scaled = stylo_scaler.transform(X_stylo_heavy[test_idx])
    heavy_emb_scaled = emb_scaler.transform(X_emb_heavy[test_idx])
    heavy_emb_pca = pca.transform(heavy_emb_scaled)
    X_heavy_final = np.hstack([heavy_stylo_scaled, heavy_emb_pca])
    heavy_preds = ensemble.predict(X_heavy_final)
    ood_heavy_f1 = f1_score(y_heavy[test_idx], heavy_preds, average='macro')
    ood_heavy_f1_scores.append(ood_heavy_f1)

    print(f"  -> OOD Light F1: {ood_light_f1:.4f} | OOD Heavy F1: {ood_heavy_f1:.4f}")

    # Save fold 1's complete fitted pipeline for the SHAP script (Issue 2 fix)
    if fold == 1:
        with open("fold1_pipeline.pkl", "wb") as f:
            pickle.dump({
                'ensemble': ensemble, 'stylo_scaler': stylo_scaler, 'emb_scaler': emb_scaler,
                'pca': pca, 'train_idx': train_idx, 'test_idx': test_idx
            }, f)
        print("  -> Saved fold 1's fitted pipeline to fold1_pipeline.pkl (for SHAP script)")

print("\n=== FINAL RESULTS (mean across 5 folds, PCA=75) ===")
print(f"Baseline (0% original) macro F1: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
print(f"OOD Light (~40% edit) macro F1:  {np.mean(ood_light_f1_scores):.4f} (+/- {np.std(ood_light_f1_scores):.4f})")
print(f"OOD Heavy (~65% edit) macro F1:  {np.mean(ood_heavy_f1_scores):.4f} (+/- {np.std(ood_heavy_f1_scores):.4f})")
print(f"\nChance baseline (4 classes): 0.2500")
print(f"Sample-to-feature ratio check: {len(train_idx)} samples / {75+8} features = {len(train_idx)/(75+8):.1f}:1")
