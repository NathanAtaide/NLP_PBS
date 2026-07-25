import streamlit as st

# Configuração da página
st.set_page_config(page_title="Meu App NLP", layout="centered")

# Título e descrição
st.title("Analisador de Textos (Modelo NLP)")
st.write("Esta é uma aplicação web construída em Python para processamento de linguagem natural.")

# Caixa de entrada para o usuário digitar o texto
texto_usuario = st.text_area("Insira o texto que deseja analisar:", height=150)

# Botão de ação
if st.button("Processar Texto"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, digite algum texto antes de processar.")
    else:
        # Aqui entrará a lógica real do seu modelo NLP no futuro
        # Por enquanto, faremos uma análise estrutural básica
        palavras = texto_usuario.split()
        
        st.success("Processamento concluído com sucesso!")
        
        # Exibindo os resultados em colunas
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total de Palavras", value=len(palavras))
        with col2:
            st.metric(label="Total de Caracteres", value=len(texto_usuario))
        
        st.subheader("Texto original em maiúsculas:")
        st.info(texto_usuario.upper())