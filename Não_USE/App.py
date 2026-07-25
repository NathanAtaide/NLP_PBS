import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import re
import pandas as pd
from collections import Counter

# --- NLP Libraries ---
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textstat
from langdetect import detect, detect_langs
from deep_translator import GoogleTranslator
from transformers import pipeline # Upgraded from NLTK VADER to Transformers

# ==========================================
# 1. PAGE CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(page_title="NLP Lyrics Analyzer", page_icon="🎵", layout="wide")

# Initialize Session State to prevent the page from resetting when clicking vocabulary buttons
if "lyrics_data" not in st.session_state:
    st.session_state.lyrics_data = None
if "selected_vocab" not in st.session_state:
    st.session_state.selected_vocab = None

# ==========================================
# 2. RESOURCE INITIALIZATION & CACHING
# ==========================================
@st.cache_resource
def download_nltk_data():
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

@st.cache_resource
def load_sentiment_model():
    """
    Loads a pre-trained Transformer model for sentiment analysis.
    This model understands context much better than VADER (e.g., detecting sadness 
    even if words like 'love' are present in songs like Drivers License).
    """
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

@st.cache_resource
def init_spotify():
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=st.secrets["SPOTIFY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        st.error(f"Error authenticating Spotify: {e}")
        return None

sp = init_spotify()

# ==========================================
# 3. CUSTOM DICTIONARIES & REGEX
# ==========================================
# Expanded Dictionary for slang and idioms
SLANG_DICT = {
    "so blue": "feeling very sad or depressed",
    "blue": "sad or depressed",
    "lit": "exciting or excellent",
    "flex": "to show off",
    "ghost": "to suddenly ignore someone",
    "cap": "a lie",
    "vibes": "emotional state or atmosphere",
    "catch feelings": "start to fall in love"
}

# Expanded Regex patterns to catch standard contractions including "I'm", "'cause"
CONTRACTIONS_DICT = {
    r"(?i)\bi'm\b": "I am",
    r"(?i)(?:\b|')cause\b": "because", # Catches "cause" and "'cause"
    r"(?i)\bweren't\b": "were not",
    r"(?i)\bain't\b": "am not / are not / is not",
    r"(?i)\bdon't\b": "do not",
    r"(?i)\bcan't\b": "cannot",
    r"(?i)\bwon't\b": "will not",
    r"(?i)\bit's\b": "it is / it has",
    r"(?i)\bthat's\b": "that is",
    r"(?i)\bgonna\b": "going to",
    r"(?i)\bwanna\b": "want to",
    r"(?i)\by'all\b": "you all",
    r"(?i)\bgotta\b": "got to",
    r"(?i)\boutta\b": "out of",
    r"(?i)\btryna\b": "trying to"
}

# Dictionary to map specific meanings to extracted Phrasal Verbs
PHRASAL_VERBS_DICT = {
    "drive up": "To arrive in a vehicle.",
    "give up": "To stop trying or surrender.",
    "break down": "To lose control of emotions or stop functioning.",
    "go through": "To endure or experience something difficult.",
    "come back": "To return.",
    "walk away": "To leave a situation.",
    "look around": "To investigate one's surroundings."
}

# ==========================================
# 4. API FETCHING FUNCTIONS
# ==========================================
def extract_spotify_id(url):
    match = re.search(r"track/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None

def fetch_spotify_metadata_by_id(track_id):
    try:
        track_info = sp.track(track_id)
        return track_info['name'], track_info['artists'][0]['name']
    except Exception:
        return None, None

def search_spotify_by_name(query):
    try:
        results = sp.search(q=query, type='track', limit=1)
        tracks = results['tracks']['items']
        if tracks:
            return tracks[0]['name'], tracks[0]['artists'][0]['name']
        return None, None
    except Exception:
        return None, None

def fetch_lyrics_lrclib(song_name, artist_name):
    url = "https://lrclib.net/api/get"
    params = {"track_name": song_name, "artist_name": artist_name}
    headers = {"User-Agent": "NLP_Streamlit_App/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("plainLyrics")
        return None
    except Exception as e:
        return None

# ==========================================
# 5. NLP PROCESSING FUNCTIONS
# ==========================================
def calculate_cefr_level(text):
    grade = textstat.flesch_kincaid_grade(text)
    if grade <= 4: return "A1 (Beginner)"
    elif grade <= 6: return "A2 (Elementary)"
    elif grade <= 8: return "B1 (Intermediate)"
    elif grade <= 10: return "B2 (Upper Intermediate)"
    elif grade <= 12: return "C1 (Advanced)"
    else: return "C2 (Proficient)"

def analyze_sentiment_transformer(text):
    """Uses a Deep Learning model to classify overall sentiment."""
    # Truncate text to 512 characters to avoid exceeding Transformer token limits on long songs
    truncated_text = text[:512] 
    try:
        result = sentiment_pipeline(truncated_text)[0]
        label = result['label']
        score = result['score']
        
        if label == "POSITIVE":
            return "Positive 🟢", score
        else:
            return "Negative 🔴", score
    except Exception:
        return "Neutral ⚪", 0.0

def extract_highlights(text):
    doc = nlp(text.lower())
    highlights = []
    
    # 1. Extract Phrasal Verbs
    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "prt": 
                    phrasal_verb = f"{token.text} {child.text}"
                    # Lookup specific meaning or use fallback
                    meaning = PHRASAL_VERBS_DICT.get(phrasal_verb, "Verb + Particle combination (Context dependent)")
                    highlights.append({
                        "Type": "Phrasal Verb",
                        "Word": phrasal_verb,
                        "Meaning": meaning,
                        "Confidence": "High (spaCy Dep Parsing)",
                        "Position": token.idx
                    })
    
    # 2. Extract Slangs / Idioms
    for slang, meaning in SLANG_DICT.items():
        for match in re.finditer(rf"\b{slang}\b", text.lower()):
            highlights.append({
                "Type": "Slang / Idiom",
                "Word": slang,
                "Meaning": meaning,
                "Confidence": "High (Dictionary Match)",
                "Position": match.start()
            })

    # 3. Extract Contractions
    for pattern, meaning in CONTRACTIONS_DICT.items():
        for match in re.finditer(pattern, text.lower()):
            highlights.append({
                "Type": "Contraction",
                "Word": match.group(0),
                "Meaning": meaning,
                "Confidence": "High (Regex Match)",
                "Position": match.start()
            })
            
    return sorted(highlights, key=lambda x: x["Position"])

def process_lexical_diversity_and_freq(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    valid_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in valid_words if w not in stop_words]
    
    unique_words = len(set(valid_words))
    total_words = len(valid_words)
    diversity = (unique_words / total_words) * 100 if total_words > 0 else 0
    freq_dist = Counter(meaningful_words)
    return diversity, freq_dist, set(valid_words)

def highlight_text_html(text, highlights):
    highlighted_text = text
    sorted_highlights = sorted(highlights, key=lambda x: len(x["Word"]), reverse=True)
    
    for h in sorted_highlights:
        word = h["Word"]
        meaning = h["Meaning"]
        category = h["Type"]
        color = "#ffb7b2" if category == "Phrasal Verb" else "#b2e2f2" if category == "Contraction" else "#e2f0cb"
        
        replacement = f'<mark title="{category}: {meaning}" style="background-color: {color}; border-radius: 3px; padding: 0 2px; cursor: help;">{word}</mark>'
        highlighted_text = re.sub(rf"(?i)\b{re.escape(word)}\b", replacement, highlighted_text)
        
    return highlighted_text.replace("\n", "<br>")

# ==========================================
# 6. USER INTERFACE & STATE TRIGGERING
# ==========================================
st.title("🎵 NLP Lyrics Analyzer")
st.markdown("Extract lyrics and perform advanced **Natural Language Processing** for language learning.")

tab1, tab2 = st.tabs(["🔍 Search by Song Name", "🔗 Search by Spotify Link"])

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        search_input = st.text_input("Enter Song Name & Artist:", key="name_input")
    with col2:
        st.write("") 
        st.write("") 
        if st.button("Search Name", use_container_width=True):
            with st.spinner("Searching Spotify..."):
                extracted_name, extracted_artist = search_spotify_by_name(search_input)
                if extracted_name:
                    lyrics = fetch_lyrics_lrclib(extracted_name, extracted_artist)
                    # SAVE TO SESSION STATE
                    st.session_state.lyrics_data = {
                        "song": extracted_name,
                        "artist": extracted_artist,
                        "lyrics": lyrics
                    }
                    st.session_state.selected_vocab = None # Reset vocab on new search
                else:
                    st.error("Song not found.")

with tab2:
    col1, col2 = st.columns([4, 1])
    with col1:
        link_input = st.text_input("Enter Spotify Link:", key="link_input")
    with col2:
        st.write("") 
        st.write("") 
        if st.button("Analyze Link", use_container_width=True):
            track_id = extract_spotify_id(link_input)
            if track_id:
                with st.spinner("Fetching Metadata..."):
                    extracted_name, extracted_artist = fetch_spotify_metadata_by_id(track_id)
                    if extracted_name:
                        lyrics = fetch_lyrics_lrclib(extracted_name, extracted_artist)
                        # SAVE TO SESSION STATE
                        st.session_state.lyrics_data = {
                            "song": extracted_name,
                            "artist": extracted_artist,
                            "lyrics": lyrics
                        }
                        st.session_state.selected_vocab = None
            else:
                st.error("Invalid Spotify Link format.")

st.divider()

# ==========================================
# 7. RENDER DASHBOARD (FROM SESSION STATE)
# ==========================================
# By checking session_state, the dashboard stays visible even if the user clicks a vocab button
if st.session_state.lyrics_data:
    data = st.session_state.lyrics_data
    song_name = data["song"]
    artist_name = data["artist"]
    lyrics = data["lyrics"]
    
    st.success(f"Track Identified: **{song_name}** by **{artist_name}**")
    
    if lyrics:
        # Run NLP Functions
        cefr_level = calculate_cefr_level(lyrics)
        sentiment_label, sentiment_score = analyze_sentiment_transformer(lyrics)
        highlights = extract_highlights(lyrics)
        diversity, freq_dist, unique_words_set = process_lexical_diversity_and_freq(lyrics)
        
        try:
            primary_lang = detect(lyrics)
        except:
            primary_lang = "unknown"

        st.header("📊 NLP Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CEFR Difficulty Level", cefr_level)
        m2.metric("Overall Sentiment", sentiment_label, f"Confidence: {sentiment_score:.2f}")
        m3.metric("Lexical Diversity", f"{diversity:.1f}%")
        m4.metric("Primary Language", primary_lang.upper())
        
        st.divider()

        lyric_col, data_col = st.columns([1.5, 1.5])
        
        with lyric_col:
            st.subheader("Interactive Lyrics")
            st.caption("Hover over highlighted words to see their meaning.")
            html_lyrics = highlight_text_html(lyrics, highlights)
            st.markdown(f'<div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; font-family: sans-serif; line-height: 1.8; color: #ffffff; height: 500px; overflow-y: scroll;">{html_lyrics}</div>', unsafe_allow_html=True)
            
            st.markdown("""
            **Legend:** 
            <span style="background-color:#ffb7b2; color:black; padding:2px 5px; border-radius:3px;">Phrasal Verb</span>
            <span style="background-color:#b2e2f2; color:black; padding:2px 5px; border-radius:3px;">Contraction</span>
            <span style="background-color:#e2f0cb; color:black; padding:2px 5px; border-radius:3px;">Slang/Idiom</span>
            """, unsafe_allow_html=True)

        with data_col:
            st.subheader("Extracted Features Details")
            if highlights:
                df_highlights = pd.DataFrame(highlights).drop_duplicates(subset=['Word'])
                st.dataframe(df_highlights[['Type', 'Word', 'Meaning', 'Confidence']], use_container_width=True, hide_index=True)
            else:
                st.info("No specific idioms, contractions, or phrasal verbs found.")
            
            st.subheader("Top Word Frequencies")
            df_freq = pd.DataFrame(freq_dist.most_common(10), columns=["Word", "Frequency"])
            st.bar_chart(df_freq.set_index("Word"))

        st.divider()

        # ==========================================
        # 8. DICTIONARY BUTTONS GRID
        # ==========================================
        st.header("📖 Vocabulary Search")
        st.write("Click a word to translate it and find synonyms.")
        
        word_list = sorted(list(unique_words_set))
        
        # Create a grid of buttons. 8 buttons per row.
        cols = st.columns(8)
        for i, word in enumerate(word_list):
            with cols[i % 8]:
                # If a button is clicked, it updates the session state
                if st.button(word, key=f"btn_{word}", use_container_width=True):
                    st.session_state.selected_vocab = word

        # Translation Logic based on the clicked button
        if st.session_state.selected_vocab:
            selected_word = st.session_state.selected_vocab
            st.markdown(f"---")
            st.markdown(f"### Analysis for: **{selected_word.upper()}**")
            
            trans_col, syn_col = st.columns(2)
            with trans_col:
                st.markdown("#### 🇧🇷 Portuguese Translation")
                try:
                    translator = GoogleTranslator(source='en', target='pt')
                    translation = translator.translate(selected_word)
                    st.info(f"➔ {translation}")
                except Exception:
                    st.error("Translation service temporarily unavailable.")
            
            with syn_col:
                st.markdown("#### 🇺🇸 English Synonyms")
                synonyms = set()
                for syn in wordnet.synsets(selected_word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name().replace("_", " "))
                
                if selected_word in synonyms:
                    synonyms.remove(selected_word)
                    
                if synonyms:
                    st.success(", ".join(list(synonyms)[:8]))
                else:
                    st.warning("No standard synonyms found in WordNet.")
    else:
        st.error("Lyrics could not be found.")