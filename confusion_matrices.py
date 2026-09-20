import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

print("Loading features, embeddings, and fold 1's saved pipeline...")
df = pd.read_csv("features_extracted.csv")
embeddings = np.load("semantic_embeddings.npy")
df['emb_idx'] = range(len(df))

STYLO_COLS = ['mean_sentence_length','sentence_length_variance','ttr','hedging_frequency',
              'passive_voice_rate','comma_density','mean_word_length','flesch_kincaid']

le = LabelEncoder()
df['model_label'] = le.fit_transform(df['model'])
CLASS_NAMES = list(le.classes_)
print("Classes:", CLASS_NAMES)

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

with open("fold1_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

ensemble = pipeline['ensemble']
stylo_scaler = pipeline['stylo_scaler']
emb_scaler = pipeline['emb_scaler']
pca = pipeline['pca']
test_idx = pipeline['test_idx']

def prep_condition(stylo_raw, emb_raw, idx_subset):
    stylo_s = stylo_scaler.transform(stylo_raw[idx_subset])
    emb_s = emb_scaler.transform(emb_raw[idx_subset])
    emb_pca = pca.transform(emb_s)
    return np.hstack([stylo_s, emb_pca])

X_test_orig = prep_condition(X_stylo_orig, X_emb_orig, test_idx)
X_test_light = prep_condition(X_stylo_light, X_emb_light, test_idx)
X_test_heavy = prep_condition(X_stylo_heavy, X_emb_heavy, test_idx)

y_test_orig = y_orig[test_idx]
y_test_light = y_light[test_idx]
y_test_heavy = y_heavy[test_idx]

preds_orig = ensemble.predict(X_test_orig)
preds_light = ensemble.predict(X_test_light)
preds_heavy = ensemble.predict(X_test_heavy)

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.xlabel('Predicted Model')
    plt.ylabel('True Model')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved {filename}")
    return cm

print("\n=== Generating confusion matrices (Fold 1, PCA=75) ===")
cm_orig = plot_confusion(y_test_orig, preds_orig, "Confusion Matrix - Original (0% edit)", "confusion_original.png")
cm_light = plot_confusion(y_test_light, preds_light, "Confusion Matrix - Light Rewrite (~40% edit)", "confusion_light.png")
cm_heavy = plot_confusion(y_test_heavy, preds_heavy, "Confusion Matrix - Heavy Rewrite (~65% edit)", "confusion_heavy.png")

print("\n=== Classification Report - Original ===")
print(classification_report(y_test_orig, preds_orig, target_names=CLASS_NAMES))
print("\n=== Classification Report - Light ===")
print(classification_report(y_test_light, preds_light, target_names=CLASS_NAMES))
print("\n=== Classification Report - Heavy ===")
print(classification_report(y_test_heavy, preds_heavy, target_names=CLASS_NAMES))

# Save raw confusion matrix numbers too
for name, cm in [('original', cm_orig), ('light', cm_light), ('heavy', cm_heavy)]:
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(f"confusion_{name}.csv")

print("\nAll confusion matrices and reports saved.")
