import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Carregar os dados
df = pd.read_csv('C:/Users/lucas/Projetos 3/movies.csv')

# Análise Exploratória de Dados Inicial
st.title("Análise Exploratória de Dados Inicial")
st.write("Análise Exploratória de Dados, Leitura dos primeiros registros do dataframe:")
st.write(df.head())
st.write("Verificar informações básicas do dataframe:")
st.write(df.info())
st.write("Verificar estatísticas descritivas do dataframe:")
st.write(df.describe())
st.write("Verificar valores nulos do dataframe:")
st.write(df.isnull().sum())

# Dicionário de dados
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

# Converter o dicionário de dados em um DataFrame
data_dict_df = pd.DataFrame(list(data_dict.items()), columns=['Coluna', 'Descrição'])

st.write("Dicionário de dados:")
st.table(data_dict_df)

# Processar a coluna genres para criar a matriz de gêneros
# Preenchendo valores nulos em genres com string vazia
df['genres'] = df['genres'].fillna('')

# Dividir os gêneros em uma lista
df['genres_list'] = df['genres'].apply(lambda x: x.split('-'))

# Criar um conjunto com todos os gêneros únicos
all_genres = set(genre for sublist in df['genres_list'] for genre in sublist if genre)

# Criar colunas binárias para cada gênero
for genre in all_genres:
    df[genre] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)

# Criar o mapa de calor
st.write("Mapa de Calor de Gêneros dos Filmes:")

# Calcular a correlação entre os gêneros
genre_corr = df[list(all_genres)].corr()

# Criar o mapa de calor interativo com Plotly
fig_heatmap = px.imshow(genre_corr, 
                        labels=dict(color="Correlação"),
                        x=list(all_genres),
                        y=list(all_genres),
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Gráfico de barras com a quantidade de filmes por gênero
st.write("Quantidade de Filmes por Gênero:")

# Contar a quantidade de filmes por gênero
genre_counts = df[list(all_genres)].sum().sort_values(ascending=False)

# Criar o gráfico de barras interativo com Plotly
fig_bar = px.bar(x=genre_counts.values, 
                 y=genre_counts.index, 
                 orientation='h',
                 labels={'x': 'Quantidade de Filmes', 'y': 'Gênero'},
                 title='Quantidade de Filmes por Gênero')

st.plotly_chart(fig_bar, use_container_width=True)
