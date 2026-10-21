import gradio as gr
import pickle
import glob
import spacy
import textstat
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Load every fold pipeline that exists (fold1_pipeline.pkl ... fold5_pipeline.pkl) ---
# Run train_all_folds.py to produce all 5. If only fold1_pipeline.pkl exists yet, the
# app still works fine — it just averages across whatever is available.
FOLD_FILES = sorted(glob.glob("fold*_pipeline.pkl"))
if not FOLD_FILES:
    raise FileNotFoundError(
        "No fold*_pipeline.pkl files found next to app.py. "
        "Run train_classifier_v2.py (fold1 only) or train_all_folds.py (all 5) first."
    )

pipelines = []
for fpath in FOLD_FILES:
    with open(fpath, "rb") as f:
        pipelines.append(pickle.load(f))

print(f"Loaded {len(pipelines)} fold pipeline(s): {FOLD_FILES}")

MODEL_NAMES = ['ChatGPT', 'Claude', 'Gemini', 'Grok']

print("Loading spaCy and sentence embedder...")
nlp = spacy.load('en_core_web_sm')
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=False)

HEDGING_WORDS = {'may','might','could','perhaps','possibly','seems','seem','appears',
                  'appear','suggests','suggest','likely','probably','presumably',
                  'somewhat','arguably','apparently','tends','tend'}

# This is the exact same feature-extraction logic as extract_features.py, the script
# that built the training data. Using spaCy here (rather than an approximation) means
# the live app sees features computed identically to what the model was trained on —
# no train/inference mismatch.
def extract_stylometric_features(text):
    text = str(text).strip()
    doc = nlp(text)
    sentences = list(doc.sents)
    words = [t.text for t in doc if not t.is_punct and not t.is_space]
    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)

    sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
    mean_sent_len = sum(sent_lengths) / n_sentences
    variance = sum((x - mean_sent_len) ** 2 for x in sent_lengths) / n_sentences if n_sentences > 0 else 0

    ttr = len(set(w.lower() for w in words)) / n_words
    hedge_count = sum(1 for w in words if w.lower() in HEDGING_WORDS)
    hedge_freq = (hedge_count / n_words) * 100
    passive_sents = sum(1 for sent in sentences if any(t.dep_ == 'nsubjpass' for t in sent))
    passive_rate = passive_sents / n_sentences
    comma_density = (text.count(',') / n_words) * 100
    mean_word_len = sum(len(w) for w in words) / n_words

    try:
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
    except Exception:
        flesch_kincaid = 0.0

    return np.array([[mean_sent_len, variance, ttr, hedge_freq, passive_rate,
                       comma_density, mean_word_len, flesch_kincaid]])

def predict(text):
    if not text or len(text.strip()) < 30:
        return {"Please enter at least a few sentences of text": 1.0}

    stylo_raw = extract_stylometric_features(text)
    emb_raw = embedder.encode([text])

    all_probs = []
    for p in pipelines:
        stylo_scaled = p['stylo_scaler'].transform(stylo_raw)
        emb_scaled = p['emb_scaler'].transform(emb_raw)
        emb_pca = p['pca'].transform(emb_scaled)
        X = np.hstack([stylo_scaled, emb_pca])
        probs = p['ensemble'].predict_proba(X)[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return {MODEL_NAMES[i]: float(avg_probs[i]) for i in range(4)}

demo1 = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, placeholder="Paste a piece of text here (at least a few sentences)...",
                       label="Text to Attribute"),
    outputs=gr.Label(num_top_classes=4, label="Predicted Source Model"),
    title="Which Model Leaves the Last Fingerprint?",
    description=(
        "A stylometric + semantic attribution classifier trained to distinguish ChatGPT, Claude, "
        "Gemini, and Grok generated text. Based on undergraduate thesis research measuring how well "
        "this kind of attribution survives LLM-based rewriting. "
        f"Averaging across {len(pipelines)} independently trained cross-validation fold(s). "
        "**Note:** trained on a modest research dataset (n=761) - treat predictions as illustrative, "
        "not a certified detection tool."
    ),
    examples=[
        ["The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem-solving across numerous domains, from healthcare diagnostics to climate modeling."],
    ],
)

# --- Second, independent capability: Human / AI-original / AI-paraphrased origin classifier ---
# Loads human_fold*_pipeline.pkl, produced by running 04_build_human_capability_classifier.py
# (in human_capability_ext/) locally. Those files don't exist until that script has been run, so
# this is loaded defensively: if they're missing, the app still starts and the existing 4-way
# attribution tab above works exactly as before -- only this second tab is affected, and it
# explains why it's unavailable instead of crashing.
HUMAN_FOLD_FILES = sorted(glob.glob("human_fold*_pipeline.pkl"))
human_pipelines = []
HUMAN_CLASS_NAMES = ['ai_original', 'ai_paraphrased', 'human']  # fallback order if no pipeline loads
if HUMAN_FOLD_FILES:
    for fpath in HUMAN_FOLD_FILES:
        with open(fpath, "rb") as f:
            human_pipelines.append(pickle.load(f))
    HUMAN_CLASS_NAMES = human_pipelines[0].get('classes', HUMAN_CLASS_NAMES)
    print(f"Loaded {len(human_pipelines)} human-capability fold pipeline(s): {HUMAN_FOLD_FILES}")
else:
    print(
        "WARNING: No human_fold*_pipeline.pkl files found next to app.py. "
        "The 'Content Origin Breakdown' tab will be shown but disabled until you run "
        "04_build_human_capability_classifier.py locally (see README.md). "
        "The existing 4-way attribution tab above is unaffected."
    )

# Friendlier display names for the raw class labels the classifier was trained on.
ORIGIN_DISPLAY_NAMES = {
    'human': 'Human-written',
    'ai_original': 'AI-generated (original)',
    'ai_paraphrased': 'AI-generated (paraphrased to evade detection)',
}

def predict_origin(text):
    if not human_pipelines:
        return {
            "Not available yet -- run 04_build_human_capability_classifier.py locally first "
            "(see README.md)": 1.0
        }

    if not text or len(text.strip()) < 30:
        return {"Please enter at least a few sentences of text": 1.0}

    # Same stylometric-feature + MiniLM-embedding pipeline as predict() above -- this second
    # classifier is architecturally identical to the 4-way one, just trained on a different
    # 3-class label set (see human_capability_ext/04_build_human_capability_classifier.py).
    stylo_raw = extract_stylometric_features(text)
    emb_raw = embedder.encode([text])

    all_probs = []
    for p in human_pipelines:
        stylo_scaled = p['stylo_scaler'].transform(stylo_raw)
        emb_scaled = p['emb_scaler'].transform(emb_raw)
        emb_pca = p['pca'].transform(emb_scaled)
        X = np.hstack([stylo_scaled, emb_pca])
        probs = p['ensemble'].predict_proba(X)[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    return {ORIGIN_DISPLAY_NAMES.get(HUMAN_CLASS_NAMES[i], HUMAN_CLASS_NAMES[i]): float(avg_probs[i])
            for i in range(len(HUMAN_CLASS_NAMES))}

demo2 = gr.Interface(
    fn=predict_origin,
    inputs=gr.Textbox(lines=10, placeholder="Paste a piece of text here (at least a few sentences)...",
                       label="Text to Analyze"),
    outputs=gr.Label(num_top_classes=3, label="Content Origin Breakdown"),
    title="Content Origin Breakdown",
    description=(
        "**Experimental addition.** A second, independent classifier -- same stylometric + semantic "
        "feature pipeline and ensemble architecture as the attribution model above -- trained to tell "
        "Human-written text apart from AI-generated text, and from AI-generated text that was then "
        "paraphrased by another LLM specifically to reduce its AI-detectability.\n\n"
        "**Important:** \"AI-generated (paraphrased)\" means AI-generated-then-LLM-rewritten text -- "
        "it is NOT literally human-edited text. This model is also trained on a much smaller, "
        "differently-sourced human-text sample (~360 samples curated from public-domain/research "
        "corpora across the same 4 domains) than the core 4-way attribution study above. Treat its "
        "output as an illustrative, exploratory addition to the thesis, not a validated result on "
        "the same footing as the main attribution finding."
        + ("" if human_pipelines else "\n\n**Currently unavailable:** run "
           "`04_build_human_capability_classifier.py` locally to generate the required "
           "`human_fold*_pipeline.pkl` files, then restart this app.")
    ),
)

demo = gr.TabbedInterface(
    [demo1, demo2],
    ["Which Model Wrote This?", "Content Origin Breakdown"],
)

if __name__ == "__main__":
    demo.launch()
