import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import re

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(page_title="Analisador de Músicas com NLP", layout="centered")
st.title("🎵 Análise de Letras (NLP)")
st.write("Cole o link de uma música do Spotify para extrair a letra e realizar o processamento de linguagem natural.")

# ==========================================
# Autenticação das APIs
# ==========================================
@st.cache_resource
def iniciar_clientes_api():
    try:
        # Credenciais do Spotify
        auth_manager = SpotifyClientCredentials(
            client_id=st.secrets["SPOTIFY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIFY_CLIENT_SECRET"]
        )
        spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        
        # Credencial do Genius
        genius_client = lyricsgenius.Genius(st.secrets["GENIUS_ACCESS_TOKEN"])
        genius_client.verbose = False # Desativa logs desnecessários no terminal
        genius_client.remove_section_headers = True # Remove [Chorus], [Verse], etc.
        
        return spotify_client, genius_client
    except Exception as e:
        st.error(f"Erro ao autenticar APIs. Verifique suas chaves no secrets.toml. Detalhe: {e}")
        return None, None

sp, genius = iniciar_clientes_api()

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
    except Exception as e:
        st.error("Erro ao buscar informações no Spotify. Verifique se o link é válido.")
        return None, None

def buscar_letra_genius(nome_musica, nome_artista):
    """Busca a letra da música no Genius e exibe erros detalhados."""
    try:
        musica = genius.search_song(nome_musica, nome_artista)
        if musica:
            letra_limpa = re.sub(r'\d*Embed$', '', musica.lyrics)
            return letra_limpa
        return None
    except Exception as e:
        # Mostra o erro técnico na tela do app
        st.error(f"Erro técnico detalhado do Genius: {e}")
        # Força o erro a aparecer no terminal/log do Streamlit Cloud
        print(f"ERRO GENIUS: {e}") 
        return None

# ==========================================
# Interface do Usuário
# ==========================================
link_spotify = st.text_input("Link da música no Spotify:", placeholder="https://open.spotify.com/track/...")

if st.button("Extrair Letra e Analisar"):
    if not link_spotify:
        st.warning("Por favor, insira um link válido.")
    elif sp is None or genius is None:
        st.error("As APIs não estão configuradas corretamente.")
    else:
        with st.spinner("Extraindo informações do Spotify..."):
            track_id = extrair_id_spotify(link_spotify)
            
            if track_id:
                nome_musica, nome_artista = buscar_metadados_spotify(track_id)
                
                if nome_musica and nome_artista:
                    st.success(f"Música encontrada: **{nome_musica}** - {nome_artista}")
                    
                    with st.spinner("Buscando letra da música..."):
                        letra = buscar_letra_genius(nome_musica, nome_artista)
                        
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
                            st.error("Letra não encontrada no banco de dados do Genius.")
            else:
                st.error("Formato de link do Spotify inválido.")