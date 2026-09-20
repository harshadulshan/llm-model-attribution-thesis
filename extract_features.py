import pandas as pd
import spacy
import textstat
import nltk
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

nlp = spacy.load('en_core_web_sm')

HEDGING_WORDS = {'may','might','could','perhaps','possibly','seems','seem','appears',
                  'appear','suggests','suggest','likely','probably','presumably',
                  'somewhat','arguably','apparently','tends','tend'}

def extract_features(text):
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

    comma_count = text.count(',')
    comma_density = (comma_count / n_words) * 100

    mean_word_len = sum(len(w) for w in words) / n_words

    try:
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
    except Exception:
        flesch_kincaid = 0.0

    return {
        'mean_sentence_length': round(mean_sent_len, 4),
        'sentence_length_variance': round(variance, 4),
        'ttr': round(ttr, 4),
        'hedging_frequency': round(hedge_freq, 4),
        'passive_voice_rate': round(passive_rate, 4),
        'comma_density': round(comma_density, 4),
        'mean_word_length': round(mean_word_len, 4),
        'flesch_kincaid': round(flesch_kincaid, 4)
    }

print("Loading gate-verified dataset...")
df = pd.read_csv("master_rewritten_samples_FINAL.csv")

# Keep only samples where BOTH conditions passed the dual-metric gate
clean = df[(df['final_pass_light'] == True) & (df['final_pass_heavy'] == True)].copy()
print(f"Samples passing both gates: {len(clean)} of {len(df)}")

rows = []
print("Extracting features for all three conditions per sample...")
for idx, row in clean.iterrows():
    for condition, text_col in [('original', 'original_text'), ('light', 'light_rewrite'), ('heavy', 'heavy_rewrite')]:
        feats = extract_features(row[text_col])
        rows.append({
            'row_position': row['row_position'],
            'model': row['model'],
            'domain': row['domain'],
            'edit_condition': condition,
            'text': row[text_col],
            **feats
        })
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(clean)} samples processed...")

feature_df = pd.DataFrame(rows)
feature_df.to_csv("features_extracted.csv", index=False)

print(f"\nDone! {len(feature_df)} total rows (samples x 3 conditions) saved to features_extracted.csv")
print("\nSample counts by model and condition:")
print(feature_df.groupby(['model','edit_condition']).size())