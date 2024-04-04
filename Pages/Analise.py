import pandas as pd
import streamlit as st

df = pd.read_csv('C:/Users/lucas/Projetos 3/movies.csv')

st.write("Análise Exploratória de Dados,Leitura dos primeiros registros do dataframe:")
st.write(df.head())
st.write("Verificar informações Básicas do dataframe:")
st.write(df.info())
st.write("Verificar estatísticas descritivas do dataframe:")
st.write(df.describe())
st.write("Verificar valores nulos do dataframe:")
st.write(df.isnull().sum())

data_dict = {
    'id': 'Identificador único do filme',
    'title': 'Título do filme',
    'genres': 'Gêneros do filme',
    'original_language': 'Idioma original do filme',
    'overview': 'Sinopse do filme',
    'popularity': 'Popularidade do filme',
    'production_companies': 'Empresas de produção do filme',
    'release_date': 'Data de lançamento do filme',
    'budget': 'Orçamento do filme',
    'revenue': 'Receita do filme',
    'runtime': 'Duração do filme',
    'status': 'Status do filme',
    'tagline': 'Tagline do filme',
    'vote_average': 'Avaliação média do filme',
    'vote_count': 'Número de votos do filme',
    'credits': 'Créditos do filme',
    'keywords': 'Palavras-chave do filme',
    'poster_path': 'Caminho do pôster do filme',
    'backdrop_path': 'Caminho do backdrop do filme',
    'recommendations': 'Recomendações do filme'
}

st.write("Dicionário de dados:")
for column, description in data_dict.items():
    st.write(f'{column}: {description}')
