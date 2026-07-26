"""
nlp_engine.py
=============
All the Natural Language Processing for the Lyrics Analyzer.

Pipeline overview
-----------------
    raw lyrics
        |  clean_lyrics()          normalise apostrophes, drop [Chorus] tags
        v
    clean text  <-- every character offset in the app refers to THIS string
        |  normalize_for_parsing() length-preserving repair for spaCy
        v
    parsing text -> spaCy Doc      (offsets stay 1:1 with the clean text)
        |
        +-- extract_contractions()   regex, on the clean text
        +-- extract_phrasal_verbs()  PhraseMatcher(LEMMA) + dependency parse
        +-- extract_idioms_slang()   PhraseMatcher(LEMMA)
        v
    spans -> resolve_overlaps() -> render_highlighted_html()

Why the "length-preserving" normalisation?
------------------------------------------
Lyrics are full of `comin'` and `'cause`. spaCy tags those badly, so we repair
them *without changing the string length*:

    comin'  ->  coming     (6 chars -> 6 chars)
    'cause  ->   cause     (apostrophe becomes a space)

Because the length never changes, `token.idx` from the repaired document is
still a valid index into the original clean text, so every highlight shows the
learner exactly what the artist wrote while the parser sees valid English.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import spacy
import streamlit as st
import textstat
from spacy.matcher import PhraseMatcher
from transformers import pipeline

from lexicons import (
    CATEGORY_COLORS,
    IDIOMS,
    PHRASAL_VERBS,
    REDUCED_FORMS,
    SLANG,
    STANDARD_CONTRACTIONS,
)

# ==========================================================================
# 1. MODEL LOADING (cached once per Streamlit process)
# ==========================================================================
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


@st.cache_resource(show_spinner="Loading the spaCy language model...")
def load_spacy() -> Optional[spacy.language.Language]:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


@st.cache_resource(show_spinner="Loading the sentiment model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


@st.cache_resource(show_spinner="Loading the emotion model (first run downloads ~330 MB)...")
def load_emotion_model():
    """Optional 7-emotion classifier. Returns None if it cannot be downloaded."""
    try:
        return pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
    except Exception:
        return None


# ==========================================================================
# 2. TEXT CLEANING
# ==========================================================================
# Typographic apostrophes must become ASCII or none of the contraction
# patterns will ever fire (lrclib returns a mix of both).
_APOSTROPHE_VARIANTS = "’ʼ‘´`‛"
_APOSTROPHE_RE = re.compile(f"[{_APOSTROPHE_VARIANTS}]")

# "[Verse 1]", "(Chorus)", "[Pre-Chorus: Artist]" ... structural, not language.
_SECTION_TAG_RE = re.compile(
    r"^[ \t]*[\[\(](?:verse|chorus|pre[- ]?chorus|bridge|intro|outro|refrain|"
    r"hook|interlude|solo|instrumental|breakdown|post[- ]?chorus)"
    r"[^\]\)]*[\]\)][ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_lyrics(raw: str) -> str:
    """Normalise the lyrics once. Every offset in the app refers to the result."""
    text = unicodedata.normalize("NFKC", raw)
    text = _APOSTROPHE_RE.sub("'", text)
    text = _SECTION_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- length-preserving repairs used only for parsing ----------------------
# comin' -> coming, singin' -> singing  (group + "g" is the same length as
# group + "'", so offsets are untouched)
_G_DROPPING_RE = re.compile(r"(?<![\w'])(\w*[a-z]in)'(?![\w'])", re.IGNORECASE)

# The clipped words that start with an apostrophe. Replacing the apostrophe
# with a space lets spaCy see a normal word instead of a stray quote mark.
_CLIPPED_HEADS = (
    "cause|coz|em|im|er|til|till|bout|round|fore|nother|nuff|neath|gainst|tis|twas|sup"
)
_LEADING_APOSTROPHE_RE = re.compile(
    rf"(?<![\w'])'(?=(?:{_CLIPPED_HEADS})\b)", re.IGNORECASE
)


def normalize_for_parsing(text: str) -> str:
    """Repair informal spellings for spaCy without shifting any character index."""
    repaired = _G_DROPPING_RE.sub(lambda m: m.group(1) + "g", text)
    repaired = _LEADING_APOSTROPHE_RE.sub(" ", repaired)
    # Guard the core invariant of the whole module.
    if len(repaired) != len(text):
        return text
    return repaired


# ==========================================================================
# 3. SPAN HELPERS
# ==========================================================================
# Higher priority wins when two features cover the same characters and have
# the same length. Idioms are the most informative, so they win.
_PRIORITY = {"Idiom": 5, "Phrasal Verb": 4, "Slang": 3, "Reduced Form": 2, "Contraction": 1}


def _span(text: str, start: int, end: int, kind: str, meaning: str, source: str) -> Dict:
    return {
        "start": start,
        "end": end,
        "Type": kind,
        "Word": text[start:end],
        "Meaning": meaning,
        "Source": source,
    }


def resolve_overlaps(spans: List[Dict]) -> List[Dict]:
    """Keep the longest / most informative feature when two features collide.

    Without this, "give up" (phrasal verb) and "up" would both try to wrap the
    same characters and the generated HTML would be nested and broken.
    """
    ranked = sorted(
        spans,
        key=lambda s: (-(s["end"] - s["start"]), -_PRIORITY.get(s["Type"], 0), s["start"]),
    )
    kept: List[Dict] = []
    for candidate in ranked:
        collides = any(
            candidate["start"] < k["end"] and k["start"] < candidate["end"] for k in kept
        )
        if not collides:
            kept.append(candidate)
    return sorted(kept, key=lambda s: s["start"])


# ==========================================================================
# 4. CONTRACTIONS & REDUCED FORMS
# ==========================================================================
@st.cache_resource
def _build_contraction_matcher() -> Tuple[re.Pattern, Dict[str, Tuple[str, str]]]:
    """One alternation regex for every known contraction.

    The word boundary is `(?<![\\w'])...(?![\\w'])` instead of `\\b`.
    `\\b` is the reason the old version could never highlight `'cause` or
    `comin'`: `\\b` needs a word character next to the apostrophe, and at the
    edge of those words there is only whitespace.
    """
    lookup: Dict[str, Tuple[str, str]] = {}
    for form, meaning in STANDARD_CONTRACTIONS.items():
        lookup[form.lower()] = (meaning, "Contraction")
    for form, meaning in REDUCED_FORMS.items():
        lookup[form.lower()] = (meaning, "Reduced Form")

    # Longest first so "don't" is preferred over a hypothetical "do".
    alternatives = "|".join(re.escape(k) for k in sorted(lookup, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w'])(?:{alternatives})(?![\w'])", re.IGNORECASE)
    return pattern, lookup


# Bare "cause" meaning "because". Guarded so the noun ("the cause of it all")
# is not flagged. Each lookbehind is individually fixed-width, as Python needs.
_BARE_CAUSE_RE = re.compile(
    r"(?<![\w'])(?<!the )(?<!a )(?<!my )(?<!no )(?<!its )(?<!this )(?<!that )"
    r"(?<!good )(?<!lost )cause(?![\w'])",
    re.IGNORECASE,
)


def extract_contractions(text: str) -> List[Dict]:
    """Dictionary matches + two rule-based patterns, on the *clean* text."""
    pattern, lookup = _build_contraction_matcher()
    spans: List[Dict] = []

    for match in pattern.finditer(text):
        meaning, kind = lookup[match.group(0).lower()]
        spans.append(
            _span(text, match.start(), match.end(), kind, meaning, "Dictionary + regex")
        )

    # Rule: dropped -g in the -ing ending (comin', takin', nothin', lovin').
    for match in _G_DROPPING_RE.finditer(text):
        full_form = match.group(1) + "g"
        spans.append(
            _span(
                text,
                match.start(),
                match.end(),
                "Reduced Form",
                f'"{match.group(0)}" = "{full_form}" - the final g of the -ing '
                f"ending is dropped, which is normal in singing and casual speech.",
                "Rule (g-dropping)",
            )
        )

    # Rule: "cause" used as a conjunction, written without the apostrophe.
    for match in _BARE_CAUSE_RE.finditer(text):
        spans.append(
            _span(
                text,
                match.start(),
                match.end(),
                "Reduced Form",
                "because (short form of \"because\", usually written 'cause)",
                "Rule (clipping)",
            )
        )

    return spans


# --- contractions the dictionary cannot list -------------------------------
# Attaching 's / 're / 'll / 've / 'd to any word produces an open-ended set
# ("the rain's gone", "nothin's left", "my baby's callin'"), so those are found
# with the tagger instead of with a list.
_CLITICS = {
    "'s": "is / has",
    "'re": "are",
    "'ve": "have",
    "'ll": "will",
    "'d": "would / had",
    "'m": "am",
}


def extract_clitics_by_parse(doc, text: str) -> List[Dict]:
    """Find contracted verbs that are attached to an unlisted word.

    The important part is `token.tag_ == "POS"`: spaCy tags the possessive
    's ("my brother's car") differently from the contracted verb 's
    ("my brother's late"). Without that test, every possessive in the song
    would be mislabelled as a contraction.
    """
    _, known_forms = _build_contraction_matcher()
    spans: List[Dict] = []

    for token in doc:
        clitic = token.text.lower()
        if clitic not in _CLITICS:
            continue
        if token.tag_ == "POS":          # possessive, not a contraction
            continue
        if token.pos_ not in ("AUX", "VERB"):
            continue
        if token.i == 0:
            continue

        host = doc[token.i - 1]
        if host.idx + len(host.text) != token.idx:   # not written as one word
            continue
        if not host.text[-1:].isalpha():
            continue

        start, end = host.idx, token.idx + len(token.text)
        if text[start:end].lower() in known_forms:   # already in the dictionary
            continue

        spans.append(
            _span(
                text,
                start,
                end,
                "Contraction",
                f"{host.text} {_CLITICS[clitic]}",
                "Tagger (clitic detection)",
            )
        )
    return spans


# ==========================================================================
# 5. PHRASE MATCHERS (phrasal verbs, idioms, slang)
# ==========================================================================
_POSSESSIVES = ("my", "your", "his", "her", "their", "our", "its")


def _possessive_variants(phrase: str) -> List[str]:
    """Expand possessives so 'break my heart' also matches 'broke her heart'."""
    words = phrase.split()
    if not any(w in _POSSESSIVES for w in words):
        return [phrase]
    return [
        " ".join(p if w in _POSSESSIVES else w for w in words) for p in _POSSESSIVES
    ]


@st.cache_resource(show_spinner="Compiling the phrase matchers...")
def _build_phrase_matcher(_nlp, entries: Tuple[Tuple[str, str], ...]) -> PhraseMatcher:
    """Build a lemma-based PhraseMatcher.

    Matching on LEMMA (not on the literal text) is what makes
    `came back`, `comes back` and `coming back` all resolve to `come back`.
    That single change is why phrasal verbs now carry a real definition
    instead of the old "Verb + Particle combination" placeholder.
    """
    matcher = PhraseMatcher(_nlp.vocab, attr="LEMMA")

    # Flatten every variant into a single batch: one nlp.pipe() call over ~700
    # short strings is far cheaper than one call per dictionary entry.
    canonicals: List[str] = []
    variants: List[str] = []
    for canonical, _meaning in entries:
        for variant in _possessive_variants(canonical):
            canonicals.append(canonical)
            variants.append(variant)

    # The lemmatiser needs the tagger and the attribute ruler; the dependency
    # parser and the entity recogniser are dead weight for isolated phrases.
    docs = _nlp.pipe(variants, disable=["parser", "ner"])

    grouped: Dict[str, List] = {}
    for canonical, pattern_doc in zip(canonicals, docs):
        grouped.setdefault(canonical, []).append(pattern_doc)

    for canonical, patterns in grouped.items():
        matcher.add(canonical, patterns)
    return matcher


@st.cache_resource
def _matchers(_nlp):
    return {
        "Phrasal Verb": _build_phrase_matcher(_nlp, tuple(PHRASAL_VERBS.items())),
        "Idiom": _build_phrase_matcher(_nlp, tuple(IDIOMS.items())),
        "Slang": _build_phrase_matcher(_nlp, tuple(SLANG.items())),
    }


_MEANINGS = {"Phrasal Verb": PHRASAL_VERBS, "Idiom": IDIOMS, "Slang": SLANG}


def extract_phrases(nlp, doc, text: str) -> List[Dict]:
    """Dictionary-driven matches for phrasal verbs, idioms and slang."""
    spans: List[Dict] = []
    for kind, matcher in _matchers(nlp).items():
        for match_id, start, end in matcher(doc):
            canonical = nlp.vocab.strings[match_id]
            meaning = _MEANINGS[kind].get(canonical)
            if not meaning:
                continue
            token_span = doc[start:end]
            spans.append(
                _span(
                    text,
                    token_span.start_char,
                    token_span.end_char,
                    kind,
                    f"{meaning}  (base form: {canonical})" if kind == "Phrasal Verb" else meaning,
                    "Dictionary (lemma match)",
                )
            )
    return spans


def extract_phrasal_verbs_by_parse(doc, text: str) -> List[Dict]:
    """Catch what the dictionary cannot: separated and unlisted phrasal verbs.

    The PhraseMatcher only sees adjacent tokens, so "give it up" and
    "turn the music down" are invisible to it. The dependency parser marks the
    particle with `prt`, which lets us recover those.
    """
    spans: List[Dict] = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        for child in token.children:
            is_particle = child.dep_ == "prt"
            is_tight_prep = child.dep_ == "prep" and child.i == token.i + 1
            if not (is_particle or is_tight_prep):
                continue

            key = f"{token.lemma_.lower()} {child.lemma_.lower()}"
            meaning = PHRASAL_VERBS.get(key)
            if meaning:
                meaning = f"{meaning}  (base form: {key})"
                source = "Dependency parse + dictionary"
            else:
                if not is_particle:
                    continue  # a plain preposition, not worth flagging
                meaning = (
                    f'"{key}" - separable phrasal verb. The particle "{child.text}" '
                    f'changes the meaning of "{token.lemma_}"; work it out from the context.'
                )
                source = "Dependency parse (not in dictionary)"

            gap = child.i - token.i
            if 0 < gap <= 3:
                # "give it up" -> highlight the whole construction.
                spans.append(
                    _span(text, token.idx, child.idx + len(child.text),
                          "Phrasal Verb", meaning, source)
                )
            else:
                for tok in (token, child):
                    spans.append(
                        _span(text, tok.idx, tok.idx + len(tok.text),
                              "Phrasal Verb", meaning, source)
                    )
    return spans


# ==========================================================================
# 6. HTML RENDERING
# ==========================================================================
def render_highlighted_html(text: str, spans: List[Dict], enabled: Optional[set] = None) -> str:
    """Rebuild the lyrics in a single left-to-right pass.

    The old version ran `re.sub` once per feature over the growing HTML, so a
    later replacement could match inside an already-generated `title="..."`
    attribute and corrupt the markup. Walking the resolved spans in order makes
    that impossible, and it preserves the original capitalisation.
    """
    parts: List[str] = []
    cursor = 0
    for span in spans:
        if enabled is not None and span["Type"] not in enabled:
            continue
        if span["start"] < cursor:
            continue
        parts.append(html.escape(text[cursor:span["start"]], quote=False))
        # quote=True only for the attribute, so a meaning containing " cannot
        # break out of the title="..." and inject markup.
        tooltip = html.escape(f'{span["Type"]}: {span["Meaning"]}', quote=True)
        color = CATEGORY_COLORS.get(span["Type"], "#dddddd")
        parts.append(
            f'<mark title="{tooltip}" style="background-color:{color};color:#111;'
            f'border-radius:4px;padding:0 3px;cursor:help;">'
            f'{html.escape(span["Word"], quote=False)}</mark>'
        )
        cursor = span["end"]
    parts.append(html.escape(text[cursor:], quote=False))
    return "".join(parts).replace("\n", "<br>")


# ==========================================================================
# 7. READABILITY / CEFR
# ==========================================================================
def _restore_sentences(text: str) -> str:
    """Give every lyric line a full stop.

    Lyrics arrive with no punctuation, so `textstat` treats a whole song as one
    gigantic sentence and reports an absurd grade level. Each line is a
    prosodic unit, so treating a line as a sentence is a far better estimate.
    """
    sentences = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[-1] not in ".!?":
            stripped += "."
        sentences.append(stripped)
    return " ".join(sentences)


_CEFR_BANDS = [
    (3.0, "A1 (Beginner)"),
    (5.0, "A2 (Elementary)"),
    (7.0, "B1 (Intermediate)"),
    (9.5, "B2 (Upper Intermediate)"),
    (12.0, "C1 (Advanced)"),
]


def estimate_cefr(text: str) -> Tuple[str, Dict[str, float]]:
    """Blend sentence complexity with vocabulary difficulty."""
    prepared = _restore_sentences(text)
    words = textstat.lexicon_count(prepared, removepunct=True)
    if words < 20:
        return "n/a (too short)", {}

    grade = textstat.flesch_kincaid_grade(prepared)
    ease = textstat.flesch_reading_ease(prepared)
    hard_words = textstat.difficult_words(prepared)
    hard_ratio = hard_words / words

    # Short lyric lines pull the grade down, so nudge it with the share of
    # words outside the Dale-Chall list of familiar English words.
    adjusted = grade + (hard_ratio - 0.12) * 18

    level = "C2 (Proficient)"
    for threshold, label in _CEFR_BANDS:
        if adjusted <= threshold:
            level = label
            break

    return level, {
        "Flesch-Kincaid grade": round(grade, 1),
        "Flesch reading ease": round(ease, 1),
        "Difficult words": f"{hard_ratio * 100:.1f}%",
        "Adjusted score": round(adjusted, 1),
    }


# ==========================================================================
# 8. SENTIMENT
# ==========================================================================
def chunk_lyrics(text: str, max_words: int = 45) -> List[str]:
    """Group whole lines into short passages that fit the model comfortably."""
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        length = len(stripped.split())
        if current and count + length > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(stripped)
        count += length
    if current:
        chunks.append(" ".join(current))
    return chunks


def analyze_sentiment(text: str) -> Dict:
    """Score the whole song, not only its first 512 characters.

    The previous version did `text[:512]`, which is roughly the first verse of
    a song, and it counted *characters* while the model's limit is in *tokens*.
    A song that opens happily and ends in heartbreak was classified as happy.
    Here every passage is scored and the results are averaged, weighted by
    length, which also produces the emotional arc chart.
    """
    chunks = chunk_lyrics(text)
    if not chunks:
        return {"label": "Neutral", "emoji": "⚪", "score": 0.0, "arc": [], "chunks": []}

    model = load_sentiment_model()
    try:
        results = model(chunks, truncation=True, max_length=512)
    except Exception:
        return {"label": "Unavailable", "emoji": "⚪", "score": 0.0, "arc": [], "chunks": []}

    # Signed polarity: +1 fully positive, -1 fully negative.
    arc = [
        r["score"] if r["label"].upper().startswith("POS") else -r["score"]
        for r in results
    ]
    weights = [max(len(c.split()), 1) for c in chunks]
    overall = sum(a * w for a, w in zip(arc, weights)) / sum(weights)

    # A neutral band is essential for lyrics: a song that alternates between
    # longing and hope is genuinely mixed, not "51% positive".
    if overall > 0.25:
        label, emoji = "Positive", "🟢"
    elif overall < -0.25:
        label, emoji = "Negative", "🔴"
    else:
        label, emoji = "Mixed / Neutral", "🟡"

    positives = sum(1 for a in arc if a > 0)
    return {
        "label": label,
        "emoji": emoji,
        "score": overall,
        "arc": arc,
        "chunks": chunks,
        "positive_passages": positives,
        "negative_passages": len(arc) - positives,
    }


@st.cache_data(show_spinner=False)
def analyze_emotions(text: str) -> Optional[List[Tuple[str, float]]]:
    """Average the 7-emotion distribution over the whole song."""
    model = load_emotion_model()
    if model is None:
        return None
    chunks = chunk_lyrics(text)
    if not chunks:
        return None
    try:
        batches = model(chunks, truncation=True, max_length=512)
    except Exception:
        return None

    totals: Counter = Counter()
    for scores in batches:
        for item in scores:
            totals[item["label"]] += item["score"]
    return sorted(
        ((label, total / len(batches)) for label, total in totals.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


# ==========================================================================
# 9. VOCABULARY STATISTICS
# ==========================================================================
def lexical_profile(doc) -> Dict:
    """Frequency and diversity computed on lemmas rather than raw tokens.

    Counting lemmas means love / loves / loving are one vocabulary item, which
    is what a learner actually has to memorise.
    """
    words = [t for t in doc if t.is_alpha]
    if not words:
        return {"diversity": 0.0, "frequencies": Counter(), "vocabulary": [], "total": 0}

    surface_forms = {t.text.lower() for t in words}
    content = [t.lemma_.lower() for t in words if not t.is_stop and len(t.text) > 2]

    return {
        "diversity": len(surface_forms) / len(words) * 100,
        "frequencies": Counter(content),
        "vocabulary": sorted(set(content)),
        "total": len(words),
    }


# ==========================================================================
# 10. ORCHESTRATION
# ==========================================================================
@st.cache_data(show_spinner="Analysing the lyrics...")
def analyze_lyrics(raw_lyrics: str) -> Dict:
    """Run the full pipeline. Cached so re-renders are instant."""
    nlp = load_spacy()
    text = clean_lyrics(raw_lyrics)
    doc = nlp(normalize_for_parsing(text))

    spans = (
        extract_contractions(text)
        + extract_clitics_by_parse(doc, text)
        + extract_phrases(nlp, doc, text)
        + extract_phrasal_verbs_by_parse(doc, text)
    )
    resolved = resolve_overlaps(spans)

    level, metrics = estimate_cefr(text)
    return {
        "text": text,
        "spans": resolved,
        "counts": Counter(s["Type"] for s in resolved),
        "cefr": level,
        "readability": metrics,
        "sentiment": analyze_sentiment(text),
        "lexical": lexical_profile(doc),
    }
