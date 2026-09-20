import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

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
y_light = light_df['model_label'].values

X_stylo_heavy = heavy_df[STYLO_COLS].values
X_emb_heavy = embeddings[heavy_df['emb_idx'].values]
y_heavy = heavy_df['model_label'].values

def build_features(config, stylo_train, stylo_test, emb_train, emb_test,
                    stylo_light, stylo_heavy, emb_light, emb_heavy):
    stylo_scaler = StandardScaler().fit(stylo_train)
    emb_scaler = StandardScaler().fit(emb_train)

    stylo_train_s = stylo_scaler.transform(stylo_train)
    stylo_test_s = stylo_scaler.transform(stylo_test)
    stylo_light_s = stylo_scaler.transform(stylo_light)
    stylo_heavy_s = stylo_scaler.transform(stylo_heavy)

    emb_train_s = emb_scaler.transform(emb_train)
    emb_test_s = emb_scaler.transform(emb_test)
    emb_light_s = emb_scaler.transform(emb_light)
    emb_heavy_s = emb_scaler.transform(emb_heavy)

    pca = PCA(n_components=0.95, random_state=42).fit(emb_train_s)
    emb_train_pca = pca.transform(emb_train_s)
    emb_test_pca = pca.transform(emb_test_s)
    emb_light_pca = pca.transform(emb_light_s)
    emb_heavy_pca = pca.transform(emb_heavy_s)

    if config == 'stylo_only':
        return stylo_train_s, stylo_test_s, stylo_light_s, stylo_heavy_s
    elif config == 'semantic_only':
        return emb_train_pca, emb_test_pca, emb_light_pca, emb_heavy_pca
    else:  # combined
        train = np.hstack([stylo_train_s, emb_train_pca])
        test = np.hstack([stylo_test_s, emb_test_pca])
        light = np.hstack([stylo_light_s, emb_light_pca])
        heavy = np.hstack([stylo_heavy_s, emb_heavy_pca])
        return train, test, light, heavy

def make_classifier(use_ensemble):
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         random_state=42, eval_metric='mlogloss')
    if not use_ensemble:
        return xgb
    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), alpha=0.01, max_iter=300, random_state=42)
    return VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)], voting='soft')

CONFIGS = [
    ('1: Stylometric only (XGBoost)', 'stylo_only', False),
    ('2: Semantic only (XGBoost)', 'semantic_only', False),
    ('3: Combined (XGBoost)', 'combined', False),
    ('4: Combined (Full Ensemble - PRIMARY)', 'combined', True),
]

results = {name: {'baseline': [], 'light': [], 'heavy': []} for name, _, _ in CONFIGS}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Running Ablation Study: 4 Configurations x 5 Folds ===\n")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_stylo_orig, y_orig), 1):
    print(f"--- Fold {fold} ---")
    y_train, y_test = y_orig[train_idx], y_orig[test_idx]
    y_light_test = y_light[test_idx]
    y_heavy_test = y_heavy[test_idx]

    for name, feat_type, use_ensemble in CONFIGS:
        X_train, X_test, X_light, X_heavy = build_features(
            feat_type,
            X_stylo_orig[train_idx], X_stylo_orig[test_idx],
            X_emb_orig[train_idx], X_emb_orig[test_idx],
            X_stylo_light[test_idx], X_stylo_heavy[test_idx],
            X_emb_light[test_idx], X_emb_heavy[test_idx]
        )

        clf = make_classifier(use_ensemble)
        clf.fit(X_train, y_train)

        f1_base = f1_score(y_test, clf.predict(X_test), average='macro')
        f1_light = f1_score(y_light_test, clf.predict(X_light), average='macro')
        f1_heavy = f1_score(y_heavy_test, clf.predict(X_heavy), average='macro')

        results[name]['baseline'].append(f1_base)
        results[name]['light'].append(f1_light)
        results[name]['heavy'].append(f1_heavy)

        print(f"  {name}: baseline={f1_base:.4f} light={f1_light:.4f} heavy={f1_heavy:.4f}")

print("\n\n=== ABLATION STUDY FINAL RESULTS (mean across 5 folds) ===\n")
print(f"{'Configuration':<45} {'Baseline':<12} {'Light':<12} {'Heavy':<12}")
print("-" * 81)
for name, _, _ in CONFIGS:
    b = np.mean(results[name]['baseline'])
    l = np.mean(results[name]['light'])
    h = np.mean(results[name]['heavy'])
    print(f"{name:<45} {b:<12.4f} {l:<12.4f} {h:<12.4f}")

print("\nChance baseline (4 classes): 0.2500")
