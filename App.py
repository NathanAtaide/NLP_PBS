import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import re

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(page_title="Analisador de Músicas com NLP", layout="centered")
st.title("🎵 Análise de Letras (NLP)")
st.write("Cole o link de uma música do Spotify para extrair a letra e realizar o processamento de linguagem natural.")

# ==========================================
# Autenticação do Spotify
# ==========================================
@st.cache_resource
def iniciar_spotify():
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=st.secrets["SPOTIFY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        st.error(f"Erro ao autenticar Spotify. Verifique suas chaves. Detalhe: {e}")
        return None

sp = iniciar_spotify()

# ==========================================
# Funções Principais
# ==========================================
def extrair_id_spotify(url):
    """Extrai o ID da música a partir do link do Spotify."""
    match = re.search(r"track/([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None

def buscar_metadados_spotify(track_id):
    """Busca o nome da música e o artista principal no Spotify."""
    try:
        track_info = sp.track(track_id)
        nome_musica = track_info['name']
        nome_artista = track_info['artists'][0]['name']
        return nome_musica, nome_artista
    except Exception:
        return None, None

def buscar_letra_lrclib(nome_musica, nome_artista):
    """Busca a letra da música na API pública LRCLIB."""
    url = "https://lrclib.net/api/get"
    params = {
        "track_name": nome_musica,
        "artist_name": nome_artista
    }
    # Declarar um User-Agent é uma boa prática para evitar bloqueios em APIs gratuitas
    headers = {"User-Agent": "AppNLP_Streamlit_Python/1.0"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            dados = response.json()
            # plainLyrics traz a letra limpa, sem os tempos da legenda
            letra = dados.get("plainLyrics")
            return letra
        else:
            return None
    except Exception as e:
        st.error(f"Erro técnico na API de letras: {e}")
        return None

# ==========================================
# Interface do Usuário
# ==========================================
link_spotify = st.text_input("Link da música no Spotify:", placeholder="https://open.spotify.com/track/...")

if st.button("Extrair Letra e Analisar"):
    if not link_spotify:
        st.warning("Por favor, insira um link válido.")
    elif sp is None:
        st.error("A API do Spotify não está configurada corretamente.")
    else:
        with st.spinner("Extraindo informações do Spotify..."):
            track_id = extrair_id_spotify(link_spotify)
            
            if track_id:
                nome_musica, nome_artista = buscar_metadados_spotify(track_id)
                
                if nome_musica and nome_artista:
                    st.success(f"Música encontrada: **{nome_musica}** - {nome_artista}")
                    
                    with st.spinner("Buscando letra da música..."):
                        letra = buscar_letra_lrclib(nome_musica, nome_artista)
                        
                        if letra:
                            st.subheader("Letra Original")
                            st.text_area(label="Oculto", value=letra, height=300, label_visibility="collapsed")
                            
                            st.divider()
                            st.subheader("Processamento NLP (Em construção)")
                            
                            # ==========================================
                            # ESPAÇO PARA O SEU ALGORITMO DE NLP
                            # ==========================================
                            st.info("O texto da letra está armazenado na variável `letra`. Insira as chamadas do seu modelo (NLTK, spaCy, Transformers) aqui.")
                            
                        else:
                            st.error("Letra não encontrada no banco de dados. Tente uma música mais popular ou verifique a formatação do título.")
            else:
                st.error("Formato de link do Spotify inválido.")