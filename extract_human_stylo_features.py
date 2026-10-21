"""
extract_human_stylo_features.py
================================
Computes the SAME 8 hand-crafted stylometric features as extract_features.py (the
script that built features_extracted.csv for the existing 4-way AI-attribution
project), but for the newly curated human_samples.csv.

This step needs only spaCy (en_core_web_sm, already installed locally) and textstat
-- no network access and no sentence-embedding model -- so it runs entirely inside
this sandbox. The MiniLM semantic embeddings for these same rows are computed
separately, in 04_build_human_capability_classifier.py, which must be run in the
user's own local venv where huggingface.co is reachable.

Output: human_stylo_features.csv
    id, domain, text, word_count, source,
    mean_sentence_length, sentence_length_variance, ttr, hedging_frequency,
    passive_voice_rate, comma_density, mean_word_length, flesch_kincaid
"""

import os

import pandas as pd
import spacy
import textstat

# textstat's flesch_kincaid_grade() needs NLTK's cmudict corpus for syllable counting.
# On a fresh machine this isn't downloaded yet, and some sandboxed/proxied network
# setups block NLTK's downloader by default as an SSRF precaution (NLTK insists on
# resolving the download URL itself rather than trusting a configured proxy). Since
# github.com/raw.githubusercontent.com (where NLTK data actually lives) IS reachable
# in this project's sandbox, we explicitly opt in to the proxied download here rather
# than silently falling back to flesch_kincaid=0.0 for every row (which is what
# extract_features.py's try/except would otherwise do unnoticed).
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
import nltk
try:
    nltk.data.find("corpora/cmudict")
except LookupError:
    print("Downloading NLTK 'cmudict' (needed by textstat for Flesch-Kincaid)...")
    nltk.download("cmudict", quiet=True)

nlp = spacy.load("en_core_web_sm")

# Identical hedging-word list to extract_features.py -- kept as a literal copy
# (rather than an import) so this file matches the "reuse the EXACT same logic"
# requirement even if extract_features.py's module-level script code changes later.
HEDGING_WORDS = {'may', 'might', 'could', 'perhaps', 'possibly', 'seems', 'seem', 'appears',
                  'appear', 'suggests', 'suggest', 'likely', 'probably', 'presumably',
                  'somewhat', 'arguably', 'apparently', 'tends', 'tend'}


def extract_features(text):
    """Byte-for-byte the same feature logic as extract_features.py / app.py."""
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(here, "human_samples.csv")
    out_path = os.path.join(here, "human_stylo_features.csv")

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path)
    print(f"{len(df)} human samples loaded.")

    print("Extracting 8 stylometric features per sample (spaCy en_core_web_sm)...")
    feature_rows = []
    for idx, row in df.iterrows():
        feats = extract_features(row["text"])
        feature_rows.append(feats)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(df)} processed...")

    feat_df = pd.DataFrame(feature_rows)
    out_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out_df.to_csv(out_path, index=False)

    print(f"\nDone! Saved {len(out_df)} rows to {out_path}")
    print("\nFeature summary by domain (mean values):")
    stylo_cols = ['mean_sentence_length', 'sentence_length_variance', 'ttr', 'hedging_frequency',
                  'passive_voice_rate', 'comma_density', 'mean_word_length', 'flesch_kincaid']
    print(out_df.groupby('domain')[stylo_cols].mean().round(3))


if __name__ == "__main__":
    main()
