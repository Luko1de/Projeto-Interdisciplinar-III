import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Carregar os dados
df = pd.read_csv(r'C:\Users\eleve\Downloads/movies.csv')

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

# Remover vote_average iguais a 0 e vote_count menores que 300
vote_count_threshold = 300
df_votes = df[(df['vote_average'] > 0) & (df['vote_count'] > vote_count_threshold)]

# Distribuição das avaliações
fig_vote_average = px.histogram(df_votes, x='vote_average', nbins=20, labels={'vote_average': 'Avaliação média'}, title='Distribuição das avaliações')
st.plotly_chart(fig_vote_average, use_container_width=True)

# Distribuição da contagem de votos (vote_count)
fig_vote_count = px.histogram(df_votes, x='vote_count', nbins=20, labels={'vote_count': 'Contagem de votos'}, title='Distribuição da contagem de votos')
st.plotly_chart(fig_vote_count, use_container_width=True)

# Relação entre avaliação média e quantidade de votos
fig_vote_scatter = px.scatter(df_votes, x='vote_count', y='vote_average', labels={'vote_count': 'Contagem de votos', 'vote_average': 'Avaliação média'}, title='Relação entre avaliação média e Contagem de votos')
st.plotly_chart(fig_vote_scatter, use_container_width=True)

# Distribuição de notas por gênero
# Transformar DataFrame para long format para Plotly
df_genres = df_votes.melt(id_vars=['title', 'vote_average'], value_vars=list(all_genres), var_name='genre', value_name='presence')
df_genres = df_genres[df_genres['presence'] == 1]

# Criar box plot
fig_box_genre = px.box(df_genres, x='genre', y='vote_average', labels={'genre': 'Gênero', 'vote_average': 'Avaliação média'}, title='Distribuição de notas por gênero')
st.plotly_chart(fig_box_genre, use_container_width=True)

# distrubuição de notas por decada
vote_count_threshold = 150
df_dates = df[(df['vote_average'] > 0) & (df['vote_count'] > vote_count_threshold)]
# Converter release_date para datetime
df_dates['release_date'] = pd.to_datetime(df_dates['release_date'], errors='coerce')

# Adicionar coluna de década
df_dates['decade'] = (df_dates['release_date'].dt.year // 10) * 10

# Criar box plot
fig_box_decade = px.box(df_dates, x='decade', y='vote_average', labels={'decade': 'Década de lançamento', 'vote_average': 'Avaliação média'}, title='Distribuição de notas por década de lançamento')
st.plotly_chart(fig_box_decade, use_container_width=True)

# Filmes mais bem avaliados
st.write("Filmes mais bem avaliados (top 10):")
top_rated_movies = df_votes[['title', 'vote_count', 'vote_average']].sort_values(by='vote_average', ascending=False).head(10)
st.write(top_rated_movies)

# Filmes com maior número de votos
st.write("Filmes com maior número de votos (top 10):")
most_voted_movies = df_votes[['title', 'vote_count', 'vote_average']].sort_values(by='vote_count', ascending=False).head(10)
st.write(most_voted_movies)

# Preenchendo valores nulos em runtime 0
df['runtime'] = df['runtime'].fillna(0)
df_runtime = df[(df['runtime'] > 0)]

# Distribuição da duração dos filmes
fig_runtime = px.histogram(df_runtime, x='runtime', nbins=20, labels={'runtime': 'Duração (minutos)'}, title='Distribuição da duração dos filmes')
fig_runtime.update_traces(xbins=dict(
    start = 0,
    end = 500,
    size = 20
))
st.plotly_chart(fig_runtime, use_container_width=True)

# Filmes com maior duração
st.write("Filmes com maior duração (top 10):")
most_voted_movies = df_runtime[['title', 'runtime']].sort_values(by='runtime', ascending=False).head(10)
st.write(most_voted_movies)

#Palavras chaves mais comuns
st.write("Wordcloud das palavras chaves mais comuns")
df_keywords = df['keywords']
df.dropna(subset=['keywords'], axis=0, inplace=True)

#Wordcloud
all_keywords = "-".join(kw for kw in df_keywords)
stopwords = set(STOPWORDS)
stopwords.update(" ", "and", "of", "the")
wordcloud  = WordCloud(stopwords=stopwords, background_color='white', width=600, height=400).generate(all_keywords)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.set_axis_off()
plt.imshow(wordcloud)
st.pyplot(fig) #plotagem no streamlit
