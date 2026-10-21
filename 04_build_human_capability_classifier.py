"""
04_build_human_capability_classifier.py
========================================
RUN THIS SCRIPT ON YOUR OWN MACHINE, NOT IN THE SANDBOX.

This is the final assembly step for the new "Content Origin" capability: a 3-class
classifier that answers "is this text Human-written / AI-generated (original) /
AI-generated-then-paraphrased-to-evade-detection?", reusing the exact same
stylometric + semantic feature pipeline and ensemble architecture as the existing
4-way (ChatGPT/Claude/Gemini/Grok) attribution model.

WHY THIS HAS TO RUN LOCALLY: it needs `sentence_transformers.SentenceTransformer
('all-MiniLM-L6-v2')`, which downloads its weights from huggingface.co the first
time it's used. The development sandbox that produced everything else in this
folder (human_samples.csv, human_stylo_features.csv) has huggingface.co blocked at
the network proxy level, so the embedding step below CANNOT run there. Every other
step in this script (data loading, balancing, PCA/ensemble/CV, evaluation, saving)
has already been dry-run-verified against the real project files in the sandbox --
only `embedder.encode(...)` itself is untested end-to-end, because it can't be.

Prerequisites (run once):
    pip install -r requirements.txt
    pip install shap          # only needed for the SHAP analysis section (step 7)
    python -m spacy download en_core_web_sm   # if not already installed

What this script does, end to end:
    1. Load human_stylo_features.csv (stylometric features already computed).
    2. Compute MiniLM embeddings for the human samples (the one step that needs
       your local venv / network access to huggingface.co).
    3. Load the existing features_extracted.csv + semantic_embeddings.npy and split
       them into ai_original (edit_condition == 'original') and ai_paraphrased
       (edit_condition in {'light','heavy'}, pooled into one class).
    4. Balance the three classes (see BALANCE_RATIO comment below for why/how).
    5. Choose a PCA dimensionality for the semantic embeddings using the same
       "defensible samples-per-feature ratio" logic train_all_folds.py used for the
       original 4-way task -- NOT the same fixed number, because the sample count
       and class count are both different here.
    6. Train the same StandardScaler(+PCA)/XGBoost+RandomForest+MLP soft-voting
       ensemble via 5-fold stratified CV, random_state=42.
    7. Evaluate: per-fold macro F1, a 3x3 confusion matrix, a stylo-only /
       semantic-only / combined ablation, and a SHAP feature-importance pass on the
       ensemble's XGBoost component.
    8. Save human_fold1_pipeline.pkl ... human_fold5_pipeline.pkl (same structure as
       the existing fold*_pipeline.pkl files) so app.py can load them the same way.
    9. Print a final, thesis-ready results summary block.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42  # matches random_state=42 used throughout this project

STYLO_COLS = ['mean_sentence_length', 'sentence_length_variance', 'ttr', 'hedging_frequency',
              'passive_voice_rate', 'comma_density', 'mean_word_length', 'flesch_kincaid']


# ======================================================================================
# STEP 1-2: Human samples -- stylometric features already computed, embed them here
# ======================================================================================
def load_human_data():
    print("=" * 88)
    print("STEP 1-2: Loading human samples and computing their MiniLM embeddings")
    print("=" * 88)
    path = os.path.join(HERE, "human_stylo_features.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} human samples from human_stylo_features.csv")
    print(df.groupby("domain").size().to_string())

    from sentence_transformers import SentenceTransformer
    print("\nLoading MiniLM embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', token=False)
    print(f"Encoding {len(df)} human texts (384-dim embeddings)...")
    texts = df['text'].astype(str).tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

    # Persist so this expensive step never has to be repeated if the rest of the
    # script is re-run (e.g. while iterating on balancing/PCA choices below).
    np.save(os.path.join(HERE, "human_embeddings.npy"), embeddings)
    print(f"Saved human_embeddings.npy, shape={embeddings.shape}")

    df['label'] = 'human'
    return df, np.asarray(embeddings)


# ======================================================================================
# STEP 3: Load the EXISTING AI dataset and split into ai_original / ai_paraphrased
# ======================================================================================
def load_ai_data():
    print("\n" + "=" * 88)
    print("STEP 3: Loading existing AI-generated dataset (features_extracted.csv)")
    print("=" * 88)
    df = pd.read_csv(os.path.join(HERE, "features_extracted.csv"))
    embeddings = np.load(os.path.join(HERE, "semantic_embeddings.npy"))
    df['emb_idx'] = range(len(df))  # positional link into semantic_embeddings.npy, unchanged by any later filtering

    ai_original = df[df['edit_condition'] == 'original'].copy()
    ai_original['label'] = 'ai_original'

    ai_paraphrased = df[df['edit_condition'].isin(['light', 'heavy'])].copy()
    ai_paraphrased['label'] = 'ai_paraphrased'

    print(f"ai_original (edit_condition == 'original'): {len(ai_original)} rows")
    print(f"ai_paraphrased (edit_condition in light/heavy, pooled): {len(ai_paraphrased)} rows")
    print("  of which:")
    print(f"    light: {(ai_paraphrased['edit_condition'] == 'light').sum()}")
    print(f"    heavy: {(ai_paraphrased['edit_condition'] == 'heavy').sum()}")
    print("\nWHY light+heavy are pooled into a single 'ai_paraphrased' class rather than kept as two "
          "classes: this classifier's research question is 'does this text look human, or does it look "
          "like AI text that has been laundered through a paraphraser to dodge attribution?' -- a binary "
          "distinction between edit intensities is a DIFFERENT question (already answered by the existing "
          "4-way model's OOD-light/OOD-heavy evaluation) and would only shrink each class further for no "
          "benefit to this specific 3-way task.")
    return ai_original, ai_paraphrased, embeddings


# ======================================================================================
# STEP 4: Balance the three classes
# ======================================================================================
# WHY balance at all: human is a small, expensive-to-source, naturally-occurring class
# (~360 rows). ai_original and ai_paraphrased are 2-4x larger simply because they were
# cheap to generate synthetically. Feeding a 3-way classifier class ratios of roughly
# 1 : 2 : 4 would bias it toward the majority classes (it could get a deceptively high
# accuracy by mostly ignoring "human" and defaulting to the AI classes) -- exactly the
# kind of artifact a thesis examiner would (rightly) flag. Balanced classes matter more
# here than preserving every last synthetic AI row, so the larger classes are
# subsampled DOWN toward -- not exactly to -- the human count, per the task brief's
# "no class exceeds roughly 1.5-2x the human count" guidance.
#
# BALANCE_RATIO=1.75 sits in the middle of that 1.5-2x band: tight enough to avoid a
# heavily majority-class-biased classifier, loose enough that the AI classes aren't
# needlessly starved of the (still cheap, still available) extra rows they have. The
# human class is NEVER touched -- it's already the smallest and it's the one class
# that genuinely can't be enlarged without new data collection.
BALANCE_RATIO = 1.75


def proportional_subsample(df, strata_cols, target_n, seed):
    """
    Subsample `df` down to `target_n` rows while preserving each stratum's (defined
    by `strata_cols`, e.g. domain x model) proportional share of the class, using the
    largest-remainder (Hamilton) apportionment method -- the same method used to
    allocate legislative seats proportionally, applied here to keep the balanced
    dataset representative of the ORIGINAL data's domain/model composition rather
    than accidentally over- or under-representing any one domain or model.
    """
    if target_n >= len(df):
        return df.copy()

    groups = df.groupby(strata_cols, dropna=False)
    sizes = groups.size()
    proportions = sizes / sizes.sum()
    raw_alloc = proportions * target_n
    alloc = np.floor(raw_alloc).astype(int)
    remainder = int(target_n - alloc.sum())

    frac = (raw_alloc - alloc).sort_values(ascending=False)
    for stratum in frac.index:
        if remainder <= 0:
            break
        if alloc[stratum] < sizes[stratum]:
            alloc[stratum] += 1
            remainder -= 1

    # If any stratum's allocation still exceeds its actual size (can't happen with the
    # logic above, but guard anyway), clip and push the leftover to strata with room.
    alloc = alloc.clip(upper=sizes)
    deficit = int(target_n - alloc.sum())
    if deficit > 0:
        room = (sizes - alloc).sort_values(ascending=False)
        for stratum in room.index:
            if deficit <= 0:
                break
            add = min(deficit, room[stratum])
            alloc[stratum] += add
            deficit -= add

    rng = np.random.default_rng(seed)
    parts = []
    for stratum, n in alloc.items():
        if n <= 0:
            continue
        sub = groups.get_group(stratum)
        idx = rng.choice(sub.index.values, size=int(n), replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts).sort_index()


def balance_classes(human_df, ai_original, ai_paraphrased):
    print("\n" + "=" * 88)
    print("STEP 4: Balancing classes")
    print("=" * 88)
    n_human = len(human_df)
    cap = int(round(BALANCE_RATIO * n_human))
    print(f"Human count (untouched, smallest natural class): {n_human}")
    print(f"Balance ratio: {BALANCE_RATIO}x  ->  cap = round({BALANCE_RATIO} x {n_human}) = {cap}")

    print(f"\nRaw ai_original count: {len(ai_original)}")
    if len(ai_original) > cap:
        ai_original_bal = proportional_subsample(ai_original, ['domain', 'model'], cap, seed=RANDOM_STATE)
        print(f"  -> subsampled to {len(ai_original_bal)} (dropped {len(ai_original) - len(ai_original_bal)}), "
              f"stratified by domain x model to preserve proportional representation")
    else:
        ai_original_bal = ai_original
        print(f"  -> already <= cap, kept all {len(ai_original_bal)} rows")

    print(f"\nRaw ai_paraphrased count: {len(ai_paraphrased)}")
    if len(ai_paraphrased) > cap:
        ai_paraphrased_bal = proportional_subsample(
            ai_paraphrased, ['domain', 'model', 'edit_condition'], cap, seed=RANDOM_STATE)
        print(f"  -> subsampled to {len(ai_paraphrased_bal)} (dropped {len(ai_paraphrased) - len(ai_paraphrased_bal)}), "
              f"stratified by domain x model x edit_condition (keeps light/heavy ~50/50 and domain/model mix intact)")
    else:
        ai_paraphrased_bal = ai_paraphrased
        print(f"  -> already <= cap, kept all {len(ai_paraphrased_bal)} rows")

    print(f"\nFinal class sizes: human={n_human}, ai_original={len(ai_original_bal)}, "
          f"ai_paraphrased={len(ai_paraphrased_bal)}")
    largest, smallest = max(n_human, len(ai_original_bal), len(ai_paraphrased_bal)), \
        min(n_human, len(ai_original_bal), len(ai_paraphrased_bal))
    print(f"Largest:smallest class ratio after balancing: {largest / smallest:.2f}:1 "
          f"(target band was 1.5-2.0:1)")

    return ai_original_bal, ai_paraphrased_bal


# ======================================================================================
# STEP 5: Choose PCA dimensionality with a defensible, printed rationale
# ======================================================================================
# The original train_all_folds.py fixed N_PCA=75 to hit a specific samples-per-feature
# ratio argued for elsewhere in the thesis (Chapter 2/3, "~83-dim vector, ~11.6:1
# ratio") that was tuned for THAT script's specific n=761-per-class 4-way task. Reusing
# 75 unchanged here would be citing someone else's homework: this task has a different
# total sample count AND a different number of classes, so the "right" PCA size is
# different too.
#
# Instead, the same KIND of argument is reapplied from scratch: a samples-per-feature
# ratio. Here we use the well-known "~10 observations per predictor variable" rule of
# thumb (an EPV -- events per variable -- heuristic with a long history in applied
# statistics for keeping a model from overfitting relative to its training set size;
# see e.g. Peduzzi et al. 1996 for the logistic-regression version of this argument,
# which the same soft-voting-ensemble-of-simple-models setup here is philosophically
# consistent with). Given this fold's training-set size, we solve for the largest
# N_PCA that keeps (8 stylometric + N_PCA semantic) features at roughly a 10:1 ratio
# to the number of training samples, then round to a clean number for reporting.
SAMPLES_PER_FEATURE_TARGET = 10.0


def choose_n_pca(train_size, n_stylo=len(STYLO_COLS), target_ratio=SAMPLES_PER_FEATURE_TARGET,
                  round_to=5, min_pca=20, max_pca=350):
    raw_budget = train_size / target_ratio - n_stylo
    n_pca = int(round(raw_budget / round_to) * round_to)
    n_pca = max(min_pca, min(n_pca, max_pca))
    total_features = n_pca + n_stylo
    achieved_ratio = train_size / total_features
    return n_pca, achieved_ratio


# ======================================================================================
# STEP 6-7: Train + evaluate (5-fold CV, confusion matrix, ablation, SHAP)
# ======================================================================================
def make_ensemble():
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         random_state=RANDOM_STATE, eval_metric='mlogloss')
    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=RANDOM_STATE)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=0.01, max_iter=300, random_state=RANDOM_STATE)
    return VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)], voting='soft')


def build_dataset(human_df, human_emb, ai_original_bal, ai_paraphrased_bal, ai_emb_all):
    """Stack all three classes into one aligned (stylo_df, embeddings, labels, domain) set."""
    frames, emb_parts = [], []

    human_df = human_df.reset_index(drop=True)
    frames.append(human_df[STYLO_COLS + ['label', 'domain']])
    emb_parts.append(human_emb)

    for name, sub in [('ai_original', ai_original_bal), ('ai_paraphrased', ai_paraphrased_bal)]:
        # NOTE: index into ai_emb_all via sub['emb_idx'] (the position in the ORIGINAL,
        # unfiltered features_extracted.csv) BEFORE resetting sub's own index below.
        emb_parts.append(ai_emb_all[sub['emb_idx'].values])
        frames.append(sub[STYLO_COLS + ['label', 'domain']].reset_index(drop=True))

    stylo_df = pd.concat(frames, ignore_index=True)
    embeddings = np.vstack(emb_parts)
    return stylo_df, embeddings


def main():
    human_df, human_emb = load_human_data()
    ai_original, ai_paraphrased, ai_emb_all = load_ai_data()
    ai_original_bal, ai_paraphrased_bal = balance_classes(human_df, ai_original, ai_paraphrased)

    stylo_df, embeddings = build_dataset(human_df, human_emb, ai_original_bal, ai_paraphrased_bal, ai_emb_all)

    le = LabelEncoder()
    y = le.fit_transform(stylo_df['label'])
    print(f"\nClasses (LabelEncoder order): {dict(zip(le.classes_, range(len(le.classes_))))}")

    X_stylo_all = stylo_df[STYLO_COLS].values
    domains = stylo_df['domain'].values

    n_total = len(stylo_df)
    approx_train_size = int(round(n_total * 4 / 5))  # 5-fold CV -> ~80% train per fold
    N_PCA, achieved_ratio = choose_n_pca(approx_train_size)

    print("\n" + "=" * 88)
    print("STEP 5: PCA dimensionality")
    print("=" * 88)
    print(f"Total balanced dataset size: {n_total} -> approx. per-fold training size: {approx_train_size}")
    print(f"Target samples-per-feature ratio: {SAMPLES_PER_FEATURE_TARGET}:1 "
          f"({len(STYLO_COLS)} stylometric + N_PCA semantic features)")
    print(f"==> N_PCA = {N_PCA}  (total feature vector = {N_PCA + len(STYLO_COLS)}-dim)")
    print(f"==> Achieved samples-per-feature ratio: {achieved_ratio:.1f}:1  "
          f"<-- CITE THIS NUMBER in the thesis the same way the original 11.6:1 argument was cited")

    print("\n" + "=" * 88)
    print("STEP 6-7: 5-fold stratified CV training + evaluation")
    print("=" * 88)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_f1_scores = []
    fold1_artifacts = {}  # kept for the confusion-matrix / SHAP steps below

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_stylo_all, y), 1):
        stylo_train, stylo_test = X_stylo_all[train_idx], X_stylo_all[test_idx]
        emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        stylo_scaler = StandardScaler().fit(stylo_train)
        emb_scaler = StandardScaler().fit(emb_train)
        stylo_train_s = stylo_scaler.transform(stylo_train)
        stylo_test_s = stylo_scaler.transform(stylo_test)
        emb_train_s = emb_scaler.transform(emb_train)
        emb_test_s = emb_scaler.transform(emb_test)

        pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(emb_train_s)
        emb_train_pca = pca.transform(emb_train_s)
        emb_test_pca = pca.transform(emb_test_s)

        X_train = np.hstack([stylo_train_s, emb_train_pca])
        X_test = np.hstack([stylo_test_s, emb_test_pca])

        ensemble = make_ensemble()
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)
        fold_f1 = f1_score(y_test, preds, average='macro')
        fold_f1_scores.append(fold_f1)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"Fold {fold}: macro F1 = {fold_f1:.4f}  (PCA={N_PCA} comps, "
              f"{explained_var*100:.1f}% variance retained)")

        pipeline = {
            'ensemble': ensemble, 'stylo_scaler': stylo_scaler, 'emb_scaler': emb_scaler,
            'pca': pca, 'label_encoder': le, 'classes': le.classes_.tolist(),
            'train_idx': train_idx, 'test_idx': test_idx, 'n_pca': N_PCA,
        }
        out_path = os.path.join(HERE, f"human_fold{fold}_pipeline.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  -> Saved {os.path.basename(out_path)}")

        if fold == 1:
            fold1_artifacts = dict(
                ensemble=ensemble, stylo_scaler=stylo_scaler, emb_scaler=emb_scaler, pca=pca,
                X_test=X_test, y_test=y_test, test_idx=test_idx, train_idx=train_idx,
                stylo_test_s=stylo_test_s, emb_test_pca=emb_test_pca,
            )

    print("\n" + "-" * 88)
    print(f"CROSS-VALIDATED RESULT: macro F1 = {np.mean(fold_f1_scores):.4f} "
          f"(+/- {np.std(fold_f1_scores):.4f}) across 5 folds")
    print(f"Chance baseline (3 classes): {1/3:.4f}")
    print("-" * 88)

    # ---- Confusion matrix (fold 1, same convention as confusion_matrices.py) ----
    print("\n" + "=" * 88)
    print("Confusion matrix (fold 1 test set)")
    print("=" * 88)
    class_names = le.classes_.tolist()
    preds_fold1 = fold1_artifacts['ensemble'].predict(fold1_artifacts['X_test'])
    cm = confusion_matrix(fold1_artifacts['y_test'], preds_fold1, labels=range(len(class_names)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(HERE, "human_confusion_matrix.csv"))
    print(cm_df.to_string())
    print("\nClassification report (fold 1 test set):")
    print(classification_report(fold1_artifacts['y_test'], preds_fold1, target_names=class_names))

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix - Human / AI-original / AI-paraphrased (Fold 1)")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "human_confusion_matrix.png"), dpi=200)
        plt.close()
        print("Saved human_confusion_matrix.png")
    except ImportError:
        print("matplotlib/seaborn not installed -- skipped plotting the confusion matrix "
              "(human_confusion_matrix.csv was still saved). Run `pip install matplotlib seaborn` "
              "to get the figure.")

    # ---- Ablation study (adapted from ablation_study_v2.py) ----
    run_ablation(X_stylo_all, embeddings, y, N_PCA, class_names)

    # ---- SHAP analysis (adapted from shap_analysis_v3.py) ----
    run_shap(fold1_artifacts, class_names)

    # ---- Final thesis-ready summary ----
    print_final_summary(human_df, ai_original, ai_paraphrased, ai_original_bal, ai_paraphrased_bal,
                         N_PCA, achieved_ratio, fold_f1_scores)


def run_ablation(X_stylo_all, embeddings, y, N_PCA, class_names):
    print("\n" + "=" * 88)
    print("Ablation study: stylometric-only / semantic-only / combined (adapted from ablation_study_v2.py)")
    print("=" * 88)

    def build_features(config, stylo_train, stylo_test, emb_train, emb_test):
        stylo_scaler = StandardScaler().fit(stylo_train)
        emb_scaler = StandardScaler().fit(emb_train)
        stylo_train_s = stylo_scaler.transform(stylo_train)
        stylo_test_s = stylo_scaler.transform(stylo_test)
        emb_train_s = emb_scaler.transform(emb_train)
        emb_test_s = emb_scaler.transform(emb_test)
        pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE).fit(emb_train_s)
        emb_train_pca = pca.transform(emb_train_s)
        emb_test_pca = pca.transform(emb_test_s)
        if config == 'stylo_only':
            return stylo_train_s, stylo_test_s
        elif config == 'semantic_only':
            return emb_train_pca, emb_test_pca
        return np.hstack([stylo_train_s, emb_train_pca]), np.hstack([stylo_test_s, emb_test_pca])

    def make_classifier(use_ensemble):
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                             random_state=RANDOM_STATE, eval_metric='mlogloss')
        if not use_ensemble:
            return xgb
        rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=RANDOM_STATE)
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=0.01, max_iter=300, random_state=RANDOM_STATE)
        return VotingClassifier(estimators=[('xgb', xgb), ('rf', rf), ('mlp', mlp)], voting='soft')

    CONFIGS = [
        ('1: Stylometric only (XGBoost)', 'stylo_only', False),
        ('2: Semantic only (XGBoost)', 'semantic_only', False),
        ('3: Combined (XGBoost)', 'combined', False),
        ('4: Combined (Full Ensemble - PRIMARY)', 'combined', True),
    ]
    results = {name: [] for name, _, _ in CONFIGS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_stylo_all, y), 1):
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"--- Fold {fold} ---")
        for name, feat_type, use_ensemble in CONFIGS:
            X_train, X_test = build_features(feat_type, X_stylo_all[train_idx], X_stylo_all[test_idx],
                                              embeddings[train_idx], embeddings[test_idx])
            clf = make_classifier(use_ensemble)
            clf.fit(X_train, y_train)
            f1 = f1_score(y_test, clf.predict(X_test), average='macro')
            results[name].append(f1)
            print(f"  {name}: macro F1={f1:.4f}")

    print("\n=== ABLATION STUDY FINAL RESULTS (mean across 5 folds) ===")
    print(f"{'Configuration':<45} {'Macro F1':<12}")
    print("-" * 57)
    rows = []
    for name, _, _ in CONFIGS:
        mean_f1 = np.mean(results[name])
        print(f"{name:<45} {mean_f1:<12.4f}")
        rows.append({'config': name, 'macro_f1_mean': mean_f1, 'macro_f1_std': np.std(results[name])})
    print(f"\nChance baseline (3 classes): {1/3:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "human_ablation_results.csv"), index=False)
    print("Saved human_ablation_results.csv")


def run_shap(fold1_artifacts, class_names):
    print("\n" + "=" * 88)
    print("SHAP feature-importance analysis on fold 1's ensemble XGBoost component "
          "(adapted from shap_analysis_v3.py)")
    print("=" * 88)
    try:
        import shap
    except ImportError:
        print("The `shap` package is not installed. Run `pip install shap` and re-run this script "
              "(or just this section) to get the SHAP analysis -- every other output has already "
              "been produced and saved.")
        return

    ensemble = fold1_artifacts['ensemble']
    xgb_from_ensemble = ensemble.named_estimators_['xgb']
    X_test = fold1_artifacts['X_test']
    y_test = fold1_artifacts['y_test']

    explainer = shap.TreeExplainer(xgb_from_ensemble)
    shap_values = np.array(explainer.shap_values(X_test))

    def true_class_importance(shap_vals, y_true):
        n_samples, n_features = shap_vals.shape[0], shap_vals.shape[1]
        true_class_shap = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            true_class_shap[i] = shap_vals[i, :, y_true[i]] if shap_vals.ndim == 3 else shap_vals[i, :]
        return np.mean(np.abs(true_class_shap), axis=0), true_class_shap

    importance, raw = true_class_importance(shap_values, y_test)

    results = []
    for i, name in enumerate(STYLO_COLS):
        results.append({'feature': name, 'type': 'stylometric', 'mean_abs_shap': importance[i]})
    pca_start = len(STYLO_COLS)
    results.append({'feature': f'semantic_block_TOTAL (sum of {len(importance) - pca_start} PCA components)',
                     'type': 'semantic', 'mean_abs_shap': np.sum(importance[pca_start:])})
    results.append({'feature': 'semantic_PER-COMPONENT average', 'type': 'semantic',
                     'mean_abs_shap': np.mean(importance[pca_start:])})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(HERE, "human_shap_feature_importance.csv"), index=False)
    print(results_df.to_string(index=False))

    stylo_only = results_df[results_df['type'] == 'stylometric'].sort_values('mean_abs_shap', ascending=False)
    print("\nStylometric features ranked by importance (true-class SHAP, fold 1 test set):")
    print(stylo_only[['feature', 'mean_abs_shap']].to_string(index=False))

    try:
        import matplotlib.pyplot as plt
        stylo_X = X_test[:, :len(STYLO_COLS)]
        stylo_raw = raw[:, :len(STYLO_COLS)]
        shap.summary_plot(stylo_raw, stylo_X, feature_names=STYLO_COLS, show=False)
        plt.title("SHAP Beeswarm — Stylometric Features (Human/AI-original/AI-paraphrased, True-Class)")
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "human_shap_beeswarm.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved human_shap_beeswarm.png")
    except ImportError:
        print("matplotlib not installed -- skipped the SHAP beeswarm plot "
              "(human_shap_feature_importance.csv was still saved).")

    print("Saved human_shap_feature_importance.csv")


def print_final_summary(human_df, ai_original, ai_paraphrased, ai_original_bal, ai_paraphrased_bal,
                         N_PCA, achieved_ratio, fold_f1_scores):
    print("\n" + "=" * 88)
    print("FINAL RESULTS SUMMARY -- copy into thesis")
    print("=" * 88)
    print(f"Class definitions:")
    print(f"  human           = newly curated human-written samples (4 domains)")
    print(f"  ai_original     = existing AI-generated samples, unmodified (edit_condition == 'original')")
    print(f"  ai_paraphrased  = existing AI-generated samples after light OR heavy LLM rewriting, pooled")
    print()
    print(f"Raw class sizes before balancing: human={len(human_df)}, "
          f"ai_original={len(ai_original)}, ai_paraphrased={len(ai_paraphrased)}")
    print(f"Balanced class sizes used for training: human={len(human_df)}, "
          f"ai_original={len(ai_original_bal)}, ai_paraphrased={len(ai_paraphrased_bal)} "
          f"(balance ratio cap = {BALANCE_RATIO}x human count)")
    print()
    print(f"PCA dimensionality: {N_PCA} components "
          f"(vs. N_PCA=75 in the original 4-way task -- see step 5's printed rationale above)")
    print(f"Samples-per-feature ratio achieved: {achieved_ratio:.1f}:1")
    print()
    print(f"5-fold CV macro F1: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    print(f"Per-fold scores: {[round(s, 4) for s in fold_f1_scores]}")
    print(f"Chance baseline (3 classes): {1/3:.4f}")
    print()
    print("See also: human_confusion_matrix.csv/.png, human_ablation_results.csv, "
          "human_shap_feature_importance.csv/.png")
    print("=" * 88)


if __name__ == "__main__":
    main()
