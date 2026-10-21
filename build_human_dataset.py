"""
build_human_dataset.py
=======================
Curates a new "human-written" text sample set for the human-capability extension
(Human / AI-generated-original / AI-generated-paraphrased 3-way classifier).

The existing 4-way attribution project has ZERO human-authored text — everything in
master_clean_originals.csv is AI-generated. This script sources real, human-authored
text for the same 4 domains the existing project uses (abstracts, books, news, reddit)
from independently-licensed, network-verified-reachable corpora that were downloaded
into raw_sources/ ahead of time (this sandbox cannot reach huggingface.co, but these
sources came from pypi/github and are already on disk):

  - abstracts -> raw_sources/arxiv_abstracts/*_abs.txt   (ArXivAbsTitleDataset, GitHub)
  - books     -> raw_sources/gutenberg/*.txt             (NLTK Gutenberg corpus)
  - news      -> raw_sources/reuters/{training,test}/*   (NLTK Reuters corpus)
  - reddit    -> raw_sources/reddit_posts.csv             (luminati-io/Reddit-dataset-samples)

Design goals (matched to the existing AI dataset's per-domain scale, see docstring
table below), all sampled with random_state=42 for reproducibility, consistent with
the rest of this project:

    Domain     | target mean words | target #samples
    abstracts  | ~191               | ~90
    books      | ~219               | ~90
    news       | ~287               | ~90
    reddit     | ~158               | ~90

Every sample gets a stable id, its domain, the raw text, its word count, and a
`source` string that traces exactly which corpus/file it came from -- this is what
lets the thesis methodology section cite provenance precisely.

Output: human_samples.csv (id, domain, text, word_count, source)
"""

import glob
import html
import os
import re

import numpy as np
import pandas as pd
import spacy

RNG_SEED = 42  # matches random_state=42 used everywhere else in this project
rng = np.random.default_rng(RNG_SEED)

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_sources")

MIN_WORDS = 50   # same floor as 02_clean_originals.py's word_count >= 50 filter
MAX_WORDS = 600  # generous upper bound; existing AI domains max out around ~570

TARGETS = {
    # domain: target_mean_words, n_samples, and a weighted-sampling "spread" (a soft
    # Gaussian std used to bias sampling toward the target mean while still allowing
    # natural variance). Each spread was hand-tuned against this specific candidate
    # pool's real length distribution (see development notes below the dict) rather
    # than derived from a single generic formula, because the four raw corpora have
    # very different natural length distributions (e.g. arXiv abstracts skew short;
    # a fixed range/4 rule overshoots for some domains and undershoots for others).
    "abstracts": {"mean": 191, "n": 90, "spread": (12, 40)},  # pool skews short (median ~140);
                                                               # asymmetric kernel needed, see
                                                               # weighted_sample() docstring
    "books":     {"mean": 219, "n": 90, "spread": 55},        # nearest-target chunking (below)
    "news":      {"mean": 287, "n": 90, "spread": 30},        # pool skews short (median ~105)
    "reddit":    {"mean": 158, "n": 90, "spread": 105},       # pool ~centered near target already
}

# A lightweight, sentencizer-only spaCy pipeline (no parser/NER/tagger) is used ONLY
# for splitting long book paragraphs at sentence boundaries when chunking excerpts.
# This is deliberately NOT the full en_core_web_sm pipeline used by extract_features.py
# -- that full pipeline (with the dependency parser, needed for passive-voice detection)
# is applied separately, once, in extract_human_stylo_features.py on the final curated
# text. Using a cheap sentencizer here just for chunking keeps this script fast.
_sentencizer = spacy.blank("en")
_sentencizer.add_pipe("sentencizer")


def word_count(text):
    return len(str(text).split())


def clean_whitespace(text):
    """Collapse all internal whitespace/newlines into single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def weighted_sample(df, n, target_mean, spread, seed=RNG_SEED):
    """
    Randomly sample up to n rows from df, weighting by closeness of word_count to
    target_mean (Gaussian kernel) so the SAMPLE's length distribution is naturally
    centered near the target -- without forcing every single sample to sit exactly
    on the mean (real natural variance is intentional, see task spec).

    `spread` is either a single std (symmetric Gaussian kernel) or a (below, above)
    tuple giving a DIFFERENT std on either side of target_mean. The asymmetric form
    is used for the two domains (abstracts, news) whose raw corpora skew noticeably
    short: a symmetric kernel plateaus well below the target mean there because
    short candidates vastly outnumber long ones even after down-weighting, so a
    tighter "below" std more aggressively suppresses that abundant short end while
    a looser "above" std still allows natural spread on the (scarcer) long end.
    """
    if len(df) == 0:
        return df.copy()
    n = min(n, len(df))
    wc = df["word_count"].values.astype(float)
    if isinstance(spread, tuple):
        spread_below, spread_above = spread
        local_spread = np.where(wc < target_mean, spread_below, spread_above)
    else:
        local_spread = spread
    weights = np.exp(-0.5 * ((wc - target_mean) / local_spread) ** 2)
    weights = weights / weights.sum()
    local_rng = np.random.default_rng(seed)
    idx = local_rng.choice(df.index.values, size=n, replace=False, p=weights)
    return df.loc[idx].copy()


# ---------------------------------------------------------------------------------
# 1) ABSTRACTS -- ArXivAbsTitleDataset (*_abs.txt, one abstract per line)
# ---------------------------------------------------------------------------------
def build_abstracts():
    print("\n[abstracts] Reading raw_sources/arxiv_abstracts/*_abs.txt ...")
    rows = []
    files = sorted(glob.glob(os.path.join(RAW, "arxiv_abstracts", "*_abs.txt")))
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                text = clean_whitespace(line)
                if not text:
                    continue
                wc = word_count(text)
                if MIN_WORDS <= wc <= MAX_WORDS:
                    rows.append({"text": text, "word_count": wc,
                                 "source": f"arxiv_abstracts:{fname}"})
    pool = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[abstracts] {len(pool)} candidate abstracts in [{MIN_WORDS},{MAX_WORDS}] words "
          f"after de-dup (from {len(files)} source files)")
    cfg = TARGETS["abstracts"]
    sampled = weighted_sample(pool, cfg["n"], cfg["mean"], cfg["spread"])
    sampled["domain"] = "abstracts"
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# 2) BOOKS -- NLTK Gutenberg corpus, chunked into ~219-word paragraph-based excerpts
# ---------------------------------------------------------------------------------
# Deliberately restricted to genuine PROSE novels/short-story collections. The
# Gutenberg corpus also includes verse (blake-poems, milton-paradise, whitman-leaves),
# play scripts (shakespeare-*), and the King James Bible (verse-numbered) -- none of
# those read as ordinary paragraph prose, so mixing them in would make "books" an
# apples-to-oranges register compared to the AI-generated "books" domain (which is
# presumably prose). 11 prose sources still comfortably clears the "6-8+ different
# books" diversity requirement.
PROSE_BOOKS = [
    "austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt",
    "carroll-alice.txt", "melville-moby_dick.txt",
    "chesterton-ball.txt", "chesterton-brown.txt", "chesterton-thursday.txt",
    "edgeworth-parents.txt", "bryant-stories.txt", "burgess-busterbrown.txt",
]

PG_START_RE = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
PG_END_RE = re.compile(r"\*\*\*\s*END OF.*", re.IGNORECASE | re.DOTALL)
BRACKET_HEADER_RE = re.compile(r"^\s*\[[^\]]*\]\s*\n")


def strip_gutenberg_boilerplate(raw_text):
    """
    Strip Project Gutenberg boilerplate. Handles two cases:
      1) The classic PG legal header/footer delimited by "*** START OF ... ***" /
         "*** END OF ... ***" (not actually present in any of these files, but the
         check costs nothing and protects against a differently-sourced Gutenberg
         file being dropped in later).
      2) The NLTK Gutenberg corpus's actual format: a single short bracketed title
         line at the very top (e.g. "[Emma by Jane Austen 1816]") and NO footer at
         all (NLTK already stripped the legal boilerplate when building the corpus).
    """
    text = raw_text
    start_match = PG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = PG_END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    text = BRACKET_HEADER_RE.sub("", text, count=1)
    return text.strip()


def split_into_paragraphs(text):
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        p = clean_whitespace(p)
        # Drop empty strings and short structural noise (chapter headings, volume
        # markers, stray single words) that aren't real prose paragraphs.
        if len(p.split()) < 8:
            continue
        # Drop ALL-CAPS heading-like lines (e.g. "CHAPTER I", "VOLUME I")
        letters = re.sub(r"[^A-Za-z]", "", p)
        if letters and p.upper() == p and len(p.split()) < 15:
            continue
        paragraphs.append(p)
    return paragraphs


def chunk_paragraphs_to_excerpts(paragraphs, target_words, min_words=97, max_words=411):
    """
    Group consecutive paragraphs into excerpts targeting ~target_words each, using a
    greedy "nearest-to-target" rule: keep adding the next paragraph to the current
    excerpt as long as doing so brings its length CLOSER to target_words; once the
    next paragraph would overshoot past the target more than stopping now would
    undershoot it, close the excerpt off and start a new one with that paragraph.
    (A naive "flush as soon as buffer_wc >= target_words" rule -- tried first --
    systematically biases every excerpt to be >= target, which is why this
    nearest-distance version is used instead: it lets excerpts land on either side
    of the target, giving a length distribution actually centered near it.)

    A single paragraph that alone exceeds max_words is split at sentence
    boundaries (via spaCy's sentencizer) into ~target_words-sized pieces.
    """
    excerpts = []
    buffer_sents = []
    buffer_wc = 0

    def flush():
        nonlocal buffer_sents, buffer_wc
        if buffer_sents:
            text = clean_whitespace(" ".join(buffer_sents))
            wc = word_count(text)
            if wc >= min_words:
                excerpts.append((text, wc))
        buffer_sents = []
        buffer_wc = 0

    for para in paragraphs:
        para_wc = word_count(para)
        if para_wc > max_words:
            # Long paragraph: split at sentence boundaries into target-sized pieces.
            flush()  # close out whatever excerpt was in progress first
            doc = _sentencizer(para)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            sub_buf, sub_wc = [], 0
            for sent in sents:
                tentative = sub_wc + word_count(sent)
                if sub_buf and abs(tentative - target_words) > abs(sub_wc - target_words) \
                        and sub_wc >= min_words:
                    excerpts.append((clean_whitespace(" ".join(sub_buf)), sub_wc))
                    sub_buf, sub_wc = [], 0
                sub_buf.append(sent)
                sub_wc += word_count(sent)
            if sub_buf and sub_wc >= min_words:
                excerpts.append((clean_whitespace(" ".join(sub_buf)), sub_wc))
            continue

        tentative_wc = buffer_wc + para_wc
        if buffer_sents and abs(tentative_wc - target_words) > abs(buffer_wc - target_words):
            # Adding this paragraph would move us further from the target than
            # stopping now does -- close off the current excerpt first.
            flush()
        buffer_sents.append(para)
        buffer_wc += para_wc

    flush()

    return [(t, w) for t, w in excerpts if min_words <= w <= max_words]


def build_books():
    print("\n[books] Reading and chunking raw_sources/gutenberg/*.txt (prose only) ...")
    cfg = TARGETS["books"]
    per_book_rows = {}
    for fname in PROSE_BOOKS:
        fpath = os.path.join(RAW, "gutenberg", fname)
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
        body = strip_gutenberg_boilerplate(raw)
        paragraphs = split_into_paragraphs(body)
        excerpts = chunk_paragraphs_to_excerpts(paragraphs, target_words=cfg["mean"])
        rows = [{"text": t, "word_count": w, "source": f"nltk_gutenberg:{fname}"}
                for t, w in excerpts]
        book_df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
        per_book_rows[fname] = book_df
        print(f"[books]   {fname}: {len(paragraphs)} paragraphs -> {len(book_df)} candidate excerpts")

    # Allocate an even quota per book so no single book dominates the final sample
    # (diversity requirement: spread across many different authors/books).
    n_books = len(PROSE_BOOKS)
    base_quota, remainder = divmod(cfg["n"], n_books)
    quotas = [base_quota + (1 if i < remainder else 0) for i in range(n_books)]

    sampled_frames = []
    shortfall = 0
    for fname, quota in zip(PROSE_BOOKS, quotas):
        book_df = per_book_rows[fname]
        got = weighted_sample(book_df, quota, cfg["mean"], cfg["spread"],
                               seed=RNG_SEED + hash(fname) % 1000)
        sampled_frames.append(got)
        shortfall += max(0, quota - len(got))

    combined = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()

    # Top off any shortfall (a book with too few candidate excerpts) from the
    # remaining pooled excerpts across all books, still weighted toward the target.
    if shortfall > 0:
        used_texts = set(combined["text"]) if len(combined) else set()
        leftover_pool = pd.concat(
            [df[~df["text"].isin(used_texts)] for df in per_book_rows.values()],
            ignore_index=True,
        )
        topoff = weighted_sample(leftover_pool, shortfall, cfg["mean"], cfg["spread"],
                                  seed=RNG_SEED + 999)
        combined = pd.concat([combined, topoff], ignore_index=True)

    combined["domain"] = "books"
    print(f"[books] Sampled {len(combined)} excerpts across {n_books} books "
          f"(quota/book ~{base_quota}-{base_quota+1})")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# 3) NEWS -- NLTK Reuters corpus (each file is already one whole article)
# ---------------------------------------------------------------------------------
# Reuters wraps BOTH short stock-ticker codes (<XON>, <BP>) AND full company names
# (<Australian Associated Press Pty Ltd>) in angle brackets as an internal markup
# convention -- neither is HTML. The bracket pair itself is not meaningful content
# (a human reading the article wouldn't see angle brackets), so it's stripped, but
# the text INSIDE is real content (a company/ticker reference) and is kept in place.
ANGLE_BRACKET_RE = re.compile(r"<([^<>]{1,60})>")


def clean_reuters_text(raw_text):
    text = html.unescape(raw_text)
    text = ANGLE_BRACKET_RE.sub(r"\1", text)  # <XON> -> XON, <Some Company Ltd> -> Some Company Ltd
    text = clean_whitespace(text)
    return text


def build_news():
    print("\n[news] Reading raw_sources/reuters/{training,test}/* ...")
    rows = []
    for split in ("training", "test"):
        split_dir = os.path.join(RAW, "reuters", split)
        for fname in os.listdir(split_dir):
            fpath = os.path.join(split_dir, fname)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
            text = clean_reuters_text(raw)
            wc = word_count(text)
            if MIN_WORDS <= wc <= MAX_WORDS:
                rows.append({"text": text, "word_count": wc,
                             "source": f"nltk_reuters:{split}/{fname}"})
    pool = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    print(f"[news] {len(pool)} candidate articles in [{MIN_WORDS},{MAX_WORDS}] words after de-dup")
    cfg = TARGETS["news"]
    sampled = weighted_sample(pool, cfg["n"], cfg["mean"], cfg["spread"])
    sampled["domain"] = "news"
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# 4) REDDIT -- reddit_posts.csv, English-only filter, description (fallback: title)
# ---------------------------------------------------------------------------------
# Lightweight English-detection heuristic (no heavy new dependency, per task spec):
# a text is treated as English if (a) it isn't dominated by non-ASCII-letter
# characters (catches Cyrillic/Arabic/CJK/etc. scripts), and (b) a reasonable
# fraction of its words are common English function words/stopwords (catches
# Spanish/Portuguese/etc. text that nonetheless uses the Latin alphabet, which (a)
# alone would miss -- verified against this dataset's real "mexico"/"brasilivre"
# subreddit posts during development).
ENGLISH_STOPWORDS = set("""
the of and to in a is that it for on with as was this be are by an at from or
have has not but they you your we i he she his her their its there here what
when where who which will would can could should about into out up down over
under more than then also just only very so if no do does did been being
""".split())


def english_score(text):
    words = re.findall(r"[A-Za-z']+", str(text).lower())
    if len(words) < 5:
        return 0.0
    letters = [ch for ch in str(text) if ch.isalpha()]
    if letters:
        non_ascii_ratio = sum(1 for ch in letters if ord(ch) > 127) / len(letters)
        if non_ascii_ratio > 0.10:
            return 0.0
    return sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)


ENGLISH_SCORE_THRESHOLD = 0.08  # validated by manual inspection during development:
# clear non-English posts (Spanish/Portuguese/etc.) scored ~0.00-0.05; genuine
# English posts scored mostly 0.15-0.45. 0.08 sits safely in the gap.


# Data-quality fix: a handful (~9/731) of raw `description` values in reddit_posts.csv
# have a stray literal "false " (or "False ") token glued onto the very front of the
# real post body -- almost certainly a boolean scraper flag (e.g. an "over_18"/
# "spoiler" field) that got concatenated with no delimiter upstream in the source
# dataset. It is not part of what the human author actually wrote, so it is stripped
# rather than left in as if it were genuine text.
LEAKED_BOOL_PREFIX_RE = re.compile(r"^(?:true|false)\s+", re.IGNORECASE)


def build_reddit():
    print("\n[reddit] Reading raw_sources/reddit_posts.csv ...")
    df = pd.read_csv(os.path.join(RAW, "reddit_posts.csv"))
    n_raw = len(df)

    def pick_text(row):
        desc = clean_whitespace(row.get("description", "") or "")
        desc = LEAKED_BOOL_PREFIX_RE.sub("", desc)
        if word_count(desc) >= MIN_WORDS:
            return desc, "description"
        title = clean_whitespace(row.get("title", "") or "")
        return title, "title"

    picked = df.apply(pick_text, axis=1, result_type="expand")
    df["use_text"] = picked[0]
    df["text_field"] = picked[1]
    df["word_count"] = df["use_text"].apply(word_count)
    df["english_score"] = df["use_text"].apply(english_score)

    before_lang = ((df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)).sum()
    filtered = df[
        (df["word_count"] >= MIN_WORDS)
        & (df["word_count"] <= MAX_WORDS)
        & (df["english_score"] >= ENGLISH_SCORE_THRESHOLD)
    ].copy()
    print(f"[reddit] {n_raw} raw posts -> {before_lang} in word-count range -> "
          f"{len(filtered)} pass English-heuristic filter (threshold={ENGLISH_SCORE_THRESHOLD})")

    filtered["text"] = filtered["use_text"]
    filtered["source"] = "reddit_dataset_samples:" + filtered["text_field"]
    pool = filtered[["text", "word_count", "source"]].drop_duplicates(subset="text").reset_index(drop=True)

    cfg = TARGETS["reddit"]
    sampled = weighted_sample(pool, cfg["n"], cfg["mean"], cfg["spread"])
    sampled["domain"] = "reddit"
    return sampled.reset_index(drop=True)


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    frames = [build_abstracts(), build_books(), build_news(), build_reddit()]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    combined["id"] = [f"human_{i+1:04d}" for i in range(len(combined))]
    combined = combined[["id", "domain", "text", "word_count", "source"]]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_samples.csv")
    combined.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("HUMAN DATASET SUMMARY")
    print("=" * 78)
    print(f"Total samples: {len(combined)}  ->  saved to {out_path}\n")
    print(f"{'Domain':<12} {'n':>5} {'target_n':>9} {'mean_wc':>9} {'target_mean':>12} "
          f"{'min':>6} {'max':>6}")
    for domain, cfg in TARGETS.items():
        sub = combined[combined["domain"] == domain]
        print(f"{domain:<12} {len(sub):>5} {cfg['n']:>9} {sub['word_count'].mean():>9.1f} "
              f"{cfg['mean']:>12} {sub['word_count'].min():>6} {sub['word_count'].max():>6}")

    print("\n--- Example rows (2 per domain) ---")
    for domain in TARGETS:
        sub = combined[combined["domain"] == domain]
        for _, row in sub.head(2).iterrows():
            snippet = row["text"][:220].replace("\n", " ")
            print(f"\n[{row['id']}] domain={row['domain']} wc={row['word_count']} "
                  f"source={row['source']}")
            print(f"  \"{snippet}...\"")

    print("\nDone.")


if __name__ == "__main__":
    main()
