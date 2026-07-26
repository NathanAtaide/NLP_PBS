"""
App.py
======
NLP Lyrics Analyzer - Streamlit interface.

The app turns an English song into a language lesson: it finds the
contractions, reduced forms, phrasal verbs, idioms and slang that make
authentic lyrics hard to understand, and explains each one in context.

    App.py         -> user interface only
    nlp_engine.py  -> all the NLP (matching, readability, sentiment)
    lexicons.py    -> the reference dictionaries

Run with:  streamlit run App.py
"""

import re

import nltk
import pandas as pd
import requests
import spotipy
import streamlit as st
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, detect
from nltk.corpus import wordnet
from spotipy.oauth2 import SpotifyClientCredentials

import nlp_engine as engine
from lexicons import CATEGORY_COLORS, CATEGORY_HELP, CATEGORY_ORDER

# langdetect is randomised by default; a fixed seed makes the reported
# language reproducible for the same lyrics.
DetectorFactory.seed = 0

# ==========================================================================
# 1. PAGE CONFIGURATION & STATE
# ==========================================================================
st.set_page_config(page_title="NLP Lyrics Analyzer", page_icon="🎵", layout="wide")

st.session_state.setdefault("lyrics_data", None)
st.session_state.setdefault("selected_vocab", None)


@st.cache_resource
def download_nltk_data():
    """Only WordNet is needed now - tokenisation is handled by spaCy."""
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


download_nltk_data()

nlp = engine.load_spacy()
if nlp is None:
    st.error(
        "The spaCy English model is missing. Install it with:\n\n"
        "`python -m spacy download en_core_web_sm`"
    )
    st.stop()


@st.cache_resource
def init_spotify():
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=st.secrets["SPOTIFY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"],
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as exc:
        st.error(f"Error authenticating Spotify: {exc}")
        return None


sp = init_spotify()


# ==========================================================================
# 2. API FETCHING
# ==========================================================================
def extract_spotify_id(url):
    match = re.search(r"track/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None


def fetch_spotify_metadata_by_id(track_id):
    try:
        track_info = sp.track(track_id)
        return track_info["name"], track_info["artists"][0]["name"]
    except Exception:
        return None, None


def search_spotify_by_name(query):
    try:
        results = sp.search(q=query, type="track", limit=1)
        tracks = results["tracks"]["items"]
        if tracks:
            return tracks[0]["name"], tracks[0]["artists"][0]["name"]
        return None, None
    except Exception:
        return None, None


def fetch_lyrics_lrclib(song_name, artist_name):
    """Try the exact endpoint first, then fall back to the fuzzy search."""
    headers = {"User-Agent": "NLP_Streamlit_App/1.0"}
    try:
        response = requests.get(
            "https://lrclib.net/api/get",
            params={"track_name": song_name, "artist_name": artist_name},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            lyrics = response.json().get("plainLyrics")
            if lyrics:
                return lyrics

        response = requests.get(
            "https://lrclib.net/api/search",
            params={"track_name": song_name, "artist_name": artist_name},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            for hit in response.json():
                if hit.get("plainLyrics"):
                    return hit["plainLyrics"]
    except Exception:
        pass
    return None


def store_result(song, artist, lyrics):
    st.session_state.lyrics_data = {"song": song, "artist": artist, "lyrics": lyrics}
    st.session_state.selected_vocab = None


# ==========================================================================
# 3. SIDEBAR
# ==========================================================================
with st.sidebar:
    st.header("⚙️ Analysis options")

    active_categories = st.multiselect(
        "Highlight in the lyrics",
        options=CATEGORY_ORDER,
        default=CATEGORY_ORDER,
        help="Turn a category off to focus on one learning objective at a time.",
    )

    show_emotions = st.checkbox(
        "Detect emotions (joy, sadness, anger...)",
        value=False,
        help="Adds a second Transformer model. The first run downloads about 330 MB.",
    )

    st.divider()
    st.subheader("Legend")
    for category in CATEGORY_ORDER:
        st.markdown(
            f'<span style="background-color:{CATEGORY_COLORS[category]};color:#111;'
            f'padding:2px 8px;border-radius:4px;font-weight:600;">{category}</span>'
            f'<br><span style="font-size:0.82em;opacity:0.8;">{CATEGORY_HELP[category]}</span>',
            unsafe_allow_html=True,
        )
        st.write("")


# ==========================================================================
# 4. SEARCH
# ==========================================================================
st.title("🎵 NLP Lyrics Analyzer")
st.markdown(
    "Turn any English song into a lesson: **contractions**, **reduced forms**, "
    "**phrasal verbs**, **idioms** and **slang**, explained in context."
)

tab_name, tab_link = st.tabs(["🔍 Search by song name", "🔗 Search by Spotify link"])

with tab_name:
    col_input, col_button = st.columns([4, 1])
    with col_input:
        search_input = st.text_input("Song name & artist:", key="name_input")
    with col_button:
        st.write("")
        st.write("")
        if st.button("Search", use_container_width=True):
            with st.spinner("Searching Spotify..."):
                name, artist = search_spotify_by_name(search_input)
            if name:
                with st.spinner("Fetching lyrics..."):
                    store_result(name, artist, fetch_lyrics_lrclib(name, artist))
            else:
                st.error("Song not found.")

with tab_link:
    col_input, col_button = st.columns([4, 1])
    with col_input:
        link_input = st.text_input("Spotify track link:", key="link_input")
    with col_button:
        st.write("")
        st.write("")
        if st.button("Analyze", use_container_width=True):
            track_id = extract_spotify_id(link_input)
            if track_id:
                with st.spinner("Fetching metadata..."):
                    name, artist = fetch_spotify_metadata_by_id(track_id)
                if name:
                    with st.spinner("Fetching lyrics..."):
                        store_result(name, artist, fetch_lyrics_lrclib(name, artist))
                else:
                    st.error("Track not found on Spotify.")
            else:
                st.error("Invalid Spotify link format.")

st.divider()

# ==========================================================================
# 5. DASHBOARD
# ==========================================================================
if not st.session_state.lyrics_data:
    st.info("Search for a song above to start the analysis.")
    st.stop()

data = st.session_state.lyrics_data
song_name, artist_name, raw_lyrics = data["song"], data["artist"], data["lyrics"]

st.success(f"Track identified: **{song_name}** by **{artist_name}**")

if not raw_lyrics or not raw_lyrics.strip():
    st.error("Lyrics could not be found for this track (it may be instrumental).")
    st.stop()

result = engine.analyze_lyrics(raw_lyrics)
lyrics = result["text"]
spans = result["spans"]
sentiment = result["sentiment"]
lexical = result["lexical"]

try:
    language = detect(lyrics).upper()
except Exception:
    language = "UNKNOWN"

if language != "EN":
    st.warning(
        f"The detected language is **{language}**. The dictionaries and models "
        "of this app are built for English, so the results may be unreliable."
    )

st.header("📊 NLP dashboard")

m1, m2, m3, m4 = st.columns(4)
m1.metric("CEFR difficulty", result["cefr"])
m2.metric(
    "Overall sentiment",
    f"{sentiment['label']} {sentiment['emoji']}",
    f"polarity {sentiment['score']:+.2f}",
)
m3.metric("Lexical diversity", f"{lexical['diversity']:.1f}%")
m4.metric("Language", language)

counts = result["counts"]
count_cols = st.columns(len(CATEGORY_ORDER))
for column, category in zip(count_cols, CATEGORY_ORDER):
    column.metric(category, counts.get(category, 0))

if result["readability"]:
    with st.expander("How is the CEFR level calculated?"):
        st.write(
            "Lyrics arrive without punctuation, so each line is treated as one "
            "sentence before the readability formulas are applied. The "
            "Flesch-Kincaid grade is then adjusted by the share of words that "
            "are outside the Dale-Chall list of familiar English words."
        )
        st.table(pd.DataFrame(result["readability"].items(), columns=["Metric", "Value"]))

st.divider()

# --------------------------------------------------------------------------
# 5a. Interactive lyrics + feature table
# --------------------------------------------------------------------------
lyric_col, data_col = st.columns([1.4, 1.6])

with lyric_col:
    st.subheader("Interactive lyrics")
    st.caption("Hover over a highlighted expression to read its meaning.")
    html_lyrics = engine.render_highlighted_html(lyrics, spans, set(active_categories))
    st.markdown(
        '<div style="background-color:#1e1e1e;padding:20px;border-radius:10px;'
        'font-family:sans-serif;line-height:2;color:#ffffff;height:520px;'
        f'overflow-y:auto;">{html_lyrics}</div>',
        unsafe_allow_html=True,
    )

with data_col:
    st.subheader("Extracted features")
    if spans:
        df = pd.DataFrame(spans)[["Type", "Word", "Meaning", "Source"]]
        feature_tabs = st.tabs(["All"] + CATEGORY_ORDER)

        with feature_tabs[0]:
            st.dataframe(
                df.drop_duplicates(subset=["Type", "Word"]),
                use_container_width=True,
                hide_index=True,
                height=460,
            )

        for tab, category in zip(feature_tabs[1:], CATEGORY_ORDER):
            with tab:
                subset = df[df["Type"] == category].drop_duplicates(subset=["Word"])
                if subset.empty:
                    st.info(f"No {category.lower()} found in this song.")
                else:
                    st.caption(CATEGORY_HELP[category])
                    st.dataframe(
                        subset[["Word", "Meaning", "Source"]],
                        use_container_width=True,
                        hide_index=True,
                        height=420,
                    )
    else:
        st.info("No contractions, phrasal verbs, idioms or slang were found.")

st.divider()

# --------------------------------------------------------------------------
# 5b. Sentiment
# --------------------------------------------------------------------------
st.header("💬 Sentiment analysis")
st.caption(
    "The song is split into passages and each one is scored separately, so a "
    "song that starts hopeful and ends in heartbreak is not flattened into a "
    "single label."
)

arc_col, emotion_col = st.columns(2)

with arc_col:
    st.subheader("Emotional arc")
    if sentiment["arc"]:
        arc_df = pd.DataFrame(
            {"Polarity": sentiment["arc"]},
            index=[f"P{i}" for i in range(1, len(sentiment["arc"]) + 1)],
        )
        st.area_chart(arc_df, height=260)
        st.caption(
            f"{sentiment['positive_passages']} positive passage(s), "
            f"{sentiment['negative_passages']} negative passage(s). "
            "+1 = fully positive, -1 = fully negative."
        )

        most_negative = min(range(len(sentiment["arc"])), key=lambda i: sentiment["arc"][i])
        most_positive = max(range(len(sentiment["arc"])), key=lambda i: sentiment["arc"][i])
        st.markdown("**Saddest passage**")
        st.info(sentiment["chunks"][most_negative])
        st.markdown("**Happiest passage**")
        st.success(sentiment["chunks"][most_positive])
    else:
        st.info("Not enough text to build a sentiment arc.")

with emotion_col:
    st.subheader("Emotions")
    if show_emotions:
        with st.spinner("Classifying emotions..."):
            emotions = engine.analyze_emotions(lyrics)
        if emotions:
            emotion_df = pd.DataFrame(emotions, columns=["Emotion", "Score"]).set_index(
                "Emotion"
            )
            st.bar_chart(emotion_df, height=260)
            top_label, top_score = emotions[0]
            st.caption(
                f"Dominant emotion: **{top_label}** ({top_score * 100:.0f}% average "
                "confidence across the passages)."
            )
        else:
            st.warning("The emotion model could not be loaded.")
    else:
        st.info("Enable *Detect emotions* in the sidebar to add a 7-emotion breakdown.")

st.divider()

# --------------------------------------------------------------------------
# 5c. Vocabulary
# --------------------------------------------------------------------------
st.header("📖 Vocabulary")

freq_col, word_col = st.columns([1, 1])

with freq_col:
    st.subheader("Most frequent content words")
    frequencies = lexical["frequencies"].most_common(12)
    if frequencies:
        freq_df = pd.DataFrame(frequencies, columns=["Word", "Frequency"]).set_index("Word")
        st.bar_chart(freq_df, height=300)
        st.caption(
            "Counted by lemma, so *love / loves / loving* are a single "
            "vocabulary item, and function words are excluded."
        )
    else:
        st.info("No content words to display.")

with word_col:
    st.subheader("Word lookup")
    vocabulary = lexical["vocabulary"]
    st.caption(f"{len(vocabulary)} distinct content words in this song.")

    chosen = st.selectbox(
        "Pick a word to translate:",
        options=vocabulary,
        index=None,
        placeholder="Type to search...",
    )
    if chosen:
        st.session_state.selected_vocab = chosen

    st.write("Or pick one of the most frequent words:")
    quick_words = [word for word, _ in frequencies[:8]]
    quick_cols = st.columns(4)
    for index, word in enumerate(quick_words):
        with quick_cols[index % 4]:
            if st.button(word, key=f"quick_{word}", use_container_width=True):
                st.session_state.selected_vocab = word

if st.session_state.selected_vocab:
    selected_word = st.session_state.selected_vocab
    st.markdown(f"### Analysis of **{selected_word.upper()}**")

    trans_col, syn_col, def_col = st.columns(3)

    with trans_col:
        st.markdown("#### 🇧🇷 Portuguese")
        try:
            st.info(f"➔ {GoogleTranslator(source='en', target='pt').translate(selected_word)}")
        except Exception:
            st.error("Translation service temporarily unavailable.")

    synsets = wordnet.synsets(selected_word)

    with syn_col:
        st.markdown("#### 🇺🇸 Synonyms")
        synonyms = {
            lemma.name().replace("_", " ")
            for synset in synsets
            for lemma in synset.lemmas()
        }
        synonyms.discard(selected_word)
        if synonyms:
            st.success(", ".join(sorted(synonyms)[:8]))
        else:
            st.warning("No synonyms found in WordNet.")

    with def_col:
        st.markdown("#### 📘 Definitions")
        if synsets:
            for synset in synsets[:3]:
                st.write(f"*({synset.pos()})* {synset.definition()}")
        else:
            st.warning("No definition found in WordNet.")
